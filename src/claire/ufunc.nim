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

proc astype*[B: static[Backend], T, U](t: Tensor[B, T], typ: typedesc[U]): Tensor[B, U] {.noSideEffect.} =
  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  result.data = t.data.map(x >= x.U)

proc fmap*[B: static[Backend], T, U](t: Tensor, g: T -> U): Tensor[B, U] {.noSideEffect.} =
  result.shape = t.shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0

  result.data = newSeq[U](result.shape.product)
  var i = 0
  for val in t:
    result.data[i] = g(val)
    inc i

template makeUniversal*(func_name: untyped) =
  proc func_name*(t: Tensor): Tensor = t.fmap(func_name)
  export func_name

template makeUniversalLocal*(func_name: untyped) =
  proc func_name(t: Tensor): Tensor = t.fmap(func_name)

makeUniversal(fac)
makeUniversal(sqrt)
makeUniversal(cbrt)
makeUniversal(ln)
makeUniversal(log10)
makeUniversal(log2)
makeUniversal(exp)
makeUniversal(arccos)
makeUniversal(arcsin)
makeUniversal(arctan)
makeUniversal(cos)
makeUniversal(cosh)
makeUniversal(sinh)
makeUniversal(sin)
makeUniversal(tan)
makeUniversal(tanh)
makeUniversal(erf)
makeUniversal(erfc)
makeUniversal(lgamma)
makeUniversal(tgamma)
makeUniversal(floor)
makeUniversal(ceil)
makeUniversal(trunc)
makeUniversal(round)
makeUniversal(degToRad)
makeUniversal(radToDeg)

