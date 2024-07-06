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
        Cpu
    Tensor*[B: static[Backend]; T] = object
        dimensions: seq[int]
        strides: seq[int]
        offset: int
        data: seq[T]

template len*(t: Tensor): int = t.data.len
template shape*(t: Tensor): seq[int] = t.dimensions
template rank*(t: Tensor): int = t.dimensions.high

proc newTensor*(dim: seq[int], T: typedesc, B: static[Backend]): Tensor[B, T] =
    let strides = (dim & 1)[1..dim.len].scanr(a * b)

    result.dimensions = dim
    result.strides = strides
    result.data = newSeq[T](dim.fold(a * b))
    result.offset = 0
    result
