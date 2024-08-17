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

proc transpose*(T: Tensor): Tensor {.noSideEffect.} =
  result.shape = t.shape.reversed
  result.strides = t.strides.reversed
  result.offset = t.offset
  result.data = t.data

proc asContiguous*[B, T](t: Tensor[B, T]): Tensor[B, T] {.noSideEffect.} =
  if t.isContiguous: return t

  result.shape = t.shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0
  result.data = newSeq[T](result.shape.product)

  var i = 0
  for val in t:
    result.data[i] = val
    inc i

proc broadcast*[B, T](t: Tensor[B, T], shape: openarray[int]): Tensor[B, T] {.noSideEffect.} =
  result = t
  assert t.rank == shape.len
  for i in 0..<result.rank:
    if result.shape[i] == 1:
      if shape[i] != 1:
        resut.shape[i] = shape[i]
        result.strides[i] = 0
    elif result.shape[i] != shape[i]:
      raise newException(ValueError, "the broadcasted size of the tensor must match existing size for non-singleton dimension")

template bc(t: Tensor, shape: openarray[int]): untyped =
  t.broadcast(shape)

proc exch_dim(t: Tensor, dim1, dim2: int): Tensor {.noSideEffect.} =
  if dim1 == dim2:
    return t
  result = t
  swap(result.strides[dim1], result.strides[dim2])
  swap(result.shape[dim1], result.shape[dim2])

proc permute*(t: Tensor, dims: varargs[int]): Tensor {.noSideEffect.} =
  var perm = @dims
  result = t
  for i, p in perm:
    if p != i and p != -1:
      var j = i
      while true:
        result = result.exch_dim(j, perm[j])
        (perm[j], j) = (-1, perm[j])
        if perm[j] == i:
          break
      perm[j] = -1
