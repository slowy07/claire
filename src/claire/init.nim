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

proc check_nested_elements(shape: seq[int], len: int) {.noSideEffect.} =
  if (shape.product != len):
    raise NewException(IndexError, "each nested sequence at the same level must have the same number of elements")

template tensor[B, T](shape: openarray[int], result: Tensor[B, T]): untyped =
  result.shape = @shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0

proc newTensor*(shape: seq[int], T: typedesc, B: static[Backend]): Tensor[B, T] {.noSideEffect.} =
  tensor(shape, result)
  result.data = newSeq[T](result.shape.product)

proc emptyTensor(shape: seq[int], T: typedesc, B: static[Backend]): Tensor[B, T] {.noSideEffect, inline.} =
  tensor(shape, result)

proc toTensor*(s: openarray, B: static[Backend]): auto {.noSideEffect.} =
  let shape = s.shape
  let data = toSeq(flatIter(s))
  when compileOption("boundChecks"): check_nested_elements(shape, data.len)
  result = newTensor(shape, type(data[0]), B)
  result.data = data

proc zeros*[T: SomeNumber](shape: openarray[int], typ: typedesc[T], B: static[Backend]): Tensor[B, T] {.noSideEffect, inline.} =
  return newTensor(shape, typ, B)

proc zeros_like*[B: static[Backend], T: SomeNumber](t: Tensor[B, T]): Tensor[B, T] {.noSideEffect, inline.} =
  return zeros(t.shape, T, B)

proc ones*[T: SomeNumber](shape: openarray[int], typ: typedesc[T], B: static[Backend]): Tensor[B, T] {.noSideEffect.} =
  tensor(shape, result)
  result.data = newSeqWith(result.shape.product, 1.T)

proc ones_like*[B: static[Backend], T: SomeNumber](t: Tensor[B, T]): Tensor[B, T] {.noSideEffect, inline.} =
  return ones(t.shape, T, B)

template randomTensorT(shape: openarray[int], max_or_range: typed): untyped =
  tensor(shape, result)
  
  result.data = newSeqWith(result.shape.product, random(max_or_range))

proc randomTensor*(shape: openarray[int], max: float, B: static[Backend]): Tensor[B, float] =
  randomTensorT(shape, max)

proc randomTensor*(shape: openarray[int], max: int, B: static[Backend]): Tensor[B, int] =
  randomTensorT(shape, max)

proc randomTensor*[T](shape: openarray[int], slice: Slice[T], B: static[Backend]): Tensor[B, T] =
  randomTensorT(shape, slice)

