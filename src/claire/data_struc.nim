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

type
    Backend* = enum
        # backend update
        Cpu
        # Cuda
        # OpenCL
        # Magma
    
    # StrideKind = enum
    # DataKind = enum
      # Dense
      # Sparse

    Tensor*[B: static[Backend]; T] = object
        shape: seq[int]
        strides: seq[int]
        offset: int
        data: seq[T]

proc len*(t: Tensor): int {.noSideEffect, inline.} = t.shape.product
template shape*(t: Tensor): seq[int] = t.shape
template strides*(t: Tensor): seq[int] = t.strides
template offset*(t: Tensor): int = t.offset
template rank*(t: Tensor): int = t.shape.len

proc shape_to_strides(shape: seq[int]): seq[int] {.noSideEffect.} =
  return (shape & 1)[1..shape.len].scanr(a * b)

proc is_C_contiguous(t: Tensor): bool {.noSideEffect.} =
  result = t.strides == t.shape.shape_to_strides
  result = result and t.strides[t.strides.high] == 1

proc is_F_contiguous(t: Tensor): bool {.noSideEffect.} =
  result = t.strides.reversed == t.shape.reversed.shape_to_strides
  result = result and t.strides[0] == 1

proc isContiguous(t: Tensor): bool {.noSideEffect.} =
  return t.is_C_contiguous or t.is_F_contiguous

proc getTransposeTarget(t: Tensor): TransposeType {.noSideEffect.} =
  if is_C_contiguous(t):
    return TransposeType.noTranspose
  elif is_F_contiguous(t):
    return TransposeType.transpose
  else:
    raise newException(ValueError, "operation not supported for this matrix,it has non-contiguus layout")

  template get_data_ptr[B, T](t: Tensor[B, T]): ptr T = unsafeAddr(t.data[0])

  proc shallowCopy*[B, T](t: Tensor[B, T]): Tensor[B, T] =
    result.shape = t.shape
    result.strides = t.strides
    result.offset = t.offset
    shallowCopy(result.data, t.data)
