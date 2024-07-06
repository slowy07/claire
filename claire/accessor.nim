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

template getIndex[B: static[Backend], T](t: Tensor[B, T], idx: varargs[int]): int =
    when CompileOption("boundChecks"):
        if idx.len != t.rank:
          raise newException(IndexError, "number of arguments: " & $(idx.len) & ", is defferent from tensor rank: " & $(t.rank))
    var real_idx = t.offset[]
    for i, j in zip(t.strides, idx):
        real_idx += i * j
    real_idx

proc `[]`*[B: static[Backend], T](t: Tensor[B, T], idx: varargs[int]): T {.noSideEffect.} =
    let real_idx = t.getIndex(idx)
    return t.data[real_idx]

proc `[]=`*[B: static[Backend], T](t: var Tensor[B, T], idx: varargs[int], val: T) {.noSideEffect.} =
    let real_idx = t.getIndex(idx)
    t.data[real_idx] = val
