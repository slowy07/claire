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
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ../data_structure, ./opencl_config.nim, ./constants, nimcl, opencl, clblast, macros
export nimcl, opencl, opencl_config, clblast

proc toClpointer*[T](p: ptr T|ptr UncheckedArray[T]): PMem {.noSideEffect, inline.} =
  cast[PMem](p)

proc toClpointer*[T](p: ClStorage[T]): PMem {.noSideEffect, inline.} =
  cast[PMem](p.Fdata)

proc toClpointer*[T](p: ClTensor[T]): PMem {.noSideEffect, inline.} =
  cast[PMem](p.storage.Fdata)

proc clMalloc*[T](size: Natural): ptr UncheckedArray[T] {.inline.} =
  cast[type result](
    buffer[T](clContext0, size)
  )

proc deallocCl*[T](p: ref[ptr UncheckedArray[T]]) {.noSideEffect.} =
  if not p[].isNil:
    check releaseMemObject p[].toClpointer

proc newClStorage*[T: SomeFloat](length: int): ClStorage[T] =
  result.Flen = length
  new(result.Fref_tracking, deallocCl)
  result.Fdata = clMalloc[T](result.Flen)
  result.Fref_tracking[] = result.Fdata

type
  ClLayoutArray = ref[ptr UncheckedArray[cint]]
  ClTensorLayout [T: SomeFloat] = object
    rank*: cint
    shape*: ClLayoutArray
    strides*: ClLayoutArray
    offset*: cint
    data*: ptr T
    len*: cint

proc layoutOnDevice*[T: SomeFloat](t: ClTensor[T]): ClTensorLayout[T] =
  result.rank = t.rank.cint
  result.offset = t.get_data_ptr
  result.len = t.size.cint

  new result.shape, deallocCl
  new result.strides, deallocCl

  var
    tmp_shape: array[MAXRANK, cint]
    tmp_strides: array[MAXRANK, cint]

  for i in 0..<t.rank:
    tmp_shape[i] = t.shape[i].cint
    tmp_strides[i] = t.strides[i].cint

  let size = t.rank * sizeof(cint)
  check enqueueWriteBuffer(
    clQueue0,
    result.shape[].toClpointer,
    CL_FALSE,
    0,
    size,
    addr tmp_shape[0],
    0, nil, nil
  )

