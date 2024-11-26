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

import ../claire_third/dynamic_stack_array, ../claire_third/tensor/datatype, nimblas, std/[complex]

export nimblas.OrderType, complex
export datatype, dynamic_stack_array

type
  CudaStorage*[T: SomeFloat] = object
    Flen*: int
    Fdata*: ptr UncheckedArray[T]
    Fref_tracking*: ref[ptr UncheckedArray[T]]

  CudaTensor*[T: SomeFloat] = object
    shape*: Metadata
    strides*: Metadata
    offset*: int
    storage*: CudaStorage[T]
  
  ClStorage*[T: SomeFloat] = object
    Flen*: int
    Fdata*: ptr UncheckedArray[T]
    Fref_tracking*: ref[ptr UncheckedArray[T]]

  ClTensor*[T: SomeFloat] = object
    shape*: Metadata
    strides*: Metadata
    offset*: int
    storage*: ClStorage[T]

  AnyTensor*[T] = Tensor[T] or CudaTensor[T] or ClTensor[T]

proc `data=`*[T](t: var Tensor[T], s: seq[T]) {.deprecated: "use copyfromraw instead".} =
  assert s.len > 0
  when T is KnownSupportsCopyMem:
    t.copyFromRaw(s[0].unsafeAddr, s.len)
  else:
    t.storage.raw_buffer = s

func rank*[T](t: CudaTensor[T] or ClTensor[T]): range[0 .. CLAIRE_MAXRANK] {.inline.} =
  t.shape.len

func size*[T](t: CudaTensor[T] or ClTensor[T]): Natural {.inline.} =  
  t.shape.product

proc shape_to_strides*(shape: Metadata, layout: OrderType = rowMajor, result: var Metadata) {.noSideEffect.} =
  var accum = 1
  result.len = shape.len
  
  if layout == rowMajor:
    for i in countdown(shape.len - 1,0):
      result[i] = accum
      accum *= shape[i]
    return

  for i in 0 ..< shape.len:
    result[i] = accum
    accum *= shape[i]
  return
