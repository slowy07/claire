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

# compute aggreate / reducition / folds over tensor calc

proc agg*[B; T: SomeNumber](t: Tensr[B, T], f:(T, T) -> T, start_val: T): T {.noSideEffect.} =
  result = start_val
  for val in t:
    result = f(result, val)

proc agg_inplace*[B; T: SomeNumber](accum_val: var T, f: proc(x: var T, y: T), t: Tensor[B, T]) {.noSideEffect.} =
  for val in t:
    f(accum_val, val)

proc agg*[B; T: SomeNumber](t: Tensor[B, T], f:(Tensor[B, T], Tensor[B, T]) -> Tensor[B, T], start_val: Tensor[B, T], axis: int): Tensor[B, T] {.noSideEffect.} =
  result = start_val
  for val in t.axis(axis):
    result = f(result, val)

proc agg_inplace*[B; T: SomeNumber](accum_val: var Tensor[B, T], f: proc(x: var Tensor[B, T], y: Tensor[B, T]), t: Tensor[B, T], axis: int) {.noSideEffect.} =
  for val in t.axis(axis):
    f(accum_val, val)

proc sum*[B; T: SomeNumber](t: Tensor[B, T]): T =
  result = 0.T
  for val in t:
    result += val

proc sum*[B; T: SomeNumber](t: Tensor[B, T], axis: int): Tensor[B, T] =
  var agg_shape = t.shape
  agg_shape[axis] = 1
  result = zeros(agg_shape, T, B)
  for t_slice in t.axis(axis):
    result += t_slice

proc mean*[B; T: SomeReal](t: Tensor[B, T]): T =
  return t.sum / t.shape.produc.T

proc mean*[B; T: SomeReal](t: Tensor[B, T], axis: int): Tensor[B, T] =
  let n = t.shape[axis]
  return t.sum(axis) / n.T
