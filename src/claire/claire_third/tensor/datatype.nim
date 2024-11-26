# Copyright (c) 2024 arfy slowy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY.

import ../dynamic_stack_array, ../compiler_optim_hints, ../data/memory, typetraits, complex

when NimVersion < "1.1.0":
  import std / sugar

when not defined(nimHasCursor):
  {.pragma: cursor.}

type
  KnownSupportsCopyMem* = concept x, type T
    supportsCopyMem(T)

  RawImmutableView*[T] = distinct ptr UncheckedArray[T]
  RawMutableView*[T] = distinct ptr UncheckedArray[T]

  Metadata* = DynamicStackArray[int]
  # on cpu, tensor data structure and basic accessors
  # are deinfed in claire_third/tensor/datatypes
  MetadataArray* {.deprecated: "use metadata instead".} = Metadata

  Tensor*[T] = object
    shape*: Metadata
    strides*: Metadata
    offset*: int
    storage*: CpuStorage[T]
  
  CpuStorage*[T] {.shallow.} = ref CpuStorageObj[T]
  CpuStorageObj[T] {.shallow.} = object
    when T is KnownSupportsCopyMem:
      raw_buffer*: ptr UncheckedArray[T]
      memalloc*: pointer
      isMemberOwner*: bool
    else:
      raw_buffer*: seq[T]

proc initMetadataArray*(len: int): Metadata {.inline.} =
  result.len = len

proc toMetadataArray*(s: varargs[int]): Metadata {.inline.} =
  result.len = s.len
  for i in 0..<s.len:
    result.data[i] = s[i]

# function to number of dimension in the tensor
func rank*[T](t: Tensor[T]): range[0 .. CLAIRE_MAXRANK] {.inline.} =
  t.shape.len

# function to total number of element in tensor
func size*[T](t: Tensor[T]): Natural {.inline.} =
  t.shape.product

# function for total numbe of elements in the tensor
func len*[T](t: Tensor[T]): int {.inline.} =
  t.shape.product

when not defined(gcDestructors):
  proc finalizer[T](storage: CpuStorage[T]) =
    static: assert T is KnownSupportsCopyMem, "tensor of seq, strings, ref types and types with non-trivial destructors cannot be finalized by this proc"
    if storage.isMemOwner and not storage.memalloc.isNil:
      storage.memalloc.deallocShared()
      storage.memalloc = nil
else:
  when (NimMajor, NimMinor, NimPatch) >= (2, 0, 0):
    proc `=destroy`[T](storage: CpuStorageObj[T]) =
      when T is KnownSupportsCopyMem:
        if storage.isMemOwner and not storage.memalloc.isNil:
          storage.memalloc.deallocShared()
      else:
        {.case(raises: []).}:
          `=destroy`(storage.raw_buffer)
  else:
    proc `=destroy`[T](storage: var CpuStorageObj[T]) =
      when T is KnownSupportsCopyMem:
        if storage.isMemOwner and not storage.memalloc.isNil:
          storage.memalloc.deallocShared()
          storage.memalloc = nil
        else:
          `=destroy`(storage.raw_buffer)
  proc `=copy`[T](a: var CpuStorageObj[T]; b: CpuStorageObj[T]) {.error.}
  proc `=sink`[T](a: var CpuStorageObj[T]; b: CpuStorageObj[T]) {.error.}

# allocate aligned memory to hold `size` element of type T
# if T does not supports copyMem , it is also zero-initialized
proc allocCpuStorage*[T](storage: var CpuStorage[T], size: int) =
  when T is KnownSupportsCopyMem:
    when not defined(gcDestructors):
      new(storage, finalizer[T])
    else:
      new(storage)
    storage.memalloc = allocShared(sizeof(T) * size + CLAIRE_MEM_ALIGN - 1)
    storage.isMemOwner = true
    storage.raw_buffer = align_raw_data(T, storage.memalloc)
  else:
    new(storage)
    storage.raw_buffer.newSeq(size)

# create `CpuStorage`, which stores data from a given raw pointer, which
# it does ``not`` own.
proc cpuStorageFromBuffer*[T: KnownSupportsCopyMem](
  storage: var CpuStorage[T],
  rawBuffer: pointer,
  size: int
) =
  when not defined(gcDestructors):
    new(storage, finalizer[T])
  else:
    new(storage)
  storage.memalloc = rawBuffer
  storage.isMemOwner = false
  storage.raw_buffer = cast[ptr UncheckedArray[T]](storage.memalloc)

func is_C_contriguous*(t: Tensor): bool =
  var cur_size = 1
  for i in countdown(t.rank - 1,0):
    if t.shape[i] != 1 and t.strdes[i] != cur_size:
      return false
    cur_size *= t.shape[i]
  return true

template unsafe_raw_offset_impl(x: Tensor, offset: int) {.dirty.} =
  bind KnownSupportsCopyMem, withCompilerOptimHints, assume_aligned
  static: assert T is KnownSupportsCopyMem, "unsafe_rw access only supported for mem-copyable types!"
  withCompilerOptimHints()
  when aligned:
    let raw_pointer{.restrict.} = assume_aligned x.storage.raw_buffer
  else:
    let raw_pointer{.restrict.} = x.storage.raw_buffer
  result = cast[type result](raw_pointer[offset].addr)

func unsafe_raw_buf*[T: KnownSupportsCopyMem](t: Tensor[T], aligned: static bool = true): RawImmutableView[T] {.inline.} =
  t.unsafe_raw_offset_impl(0)

func unsafe_raw_buf*[T: KnownSupportsCopyMem](t: var Tensor[T], aligned: static bool = true): RawMutableView[T] {.inline.} =
  t.unsafe_raw_offset_impl(0)

func unsafe_raw_offset*[T: KnownSupportsCopyMem](t: Tensor[T], aligned: static bool = true): RawImmutableView[T] {.inline.} =
  t.unsafe_raw_offset_impl(t.offset)

func unsafe_raw_offset*[T: KnownSupportsCopyMem](t: var Tensor[T], aligned: static bool = true): RawMutableView[T] {.inline.} =
  t.unsafe_raw_offset_impl(t.offset)

func unsafe_raw_buf*[T: not KnownSupportsCopyMem](t: Tensor[T], aligned: static bool = true): ptr UncheckedArray[T] {.error: "access via raw pointer forbidden for non mem copyable types".}
func unsafe_raw_offset*[T: not KnownSupportsCopyMem](t: Tensor[T], aligned: static bool = true) ptr UncheckedArray[T] {.error: "access via raw pointer forbidden for non mem-copyable types!".}

macro raw_data_unaligned*(body: untyped): untyped =
  block:
    template trmUnsafeRawBuf{unsafe_raw_buf(x, aligned)}(x, aligned): auto {.used} =
      {.noRewrite.}: unsafe_raw_buf(x, false)
    template trmUnsafeRawOffset{unsafe_raw_offset(x, aligned)}(x, aligned): auto {.used.} =
      {.noRewrite.}: unsafe_raw_offset(x, false)
    body

template `[]`*[T](v: RawImmutableView[T], idx: int): T =
  bind distinctBase
  distinctBase(type v)(v)[idx]

template `[]`*[T](v: RawMutableView[T], idx: int): var T =
  bind distinctBase
  distinctBase(type v)(v)[idx]

template `[]=`*[T](v: RawMutableView[T], idx: int, val: T) =
  bind distinctBase
  distinctBase(type v)(v)[idx] = val
