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

template matmul[B: static[Backend],T](a, b, result: Tensor[B, T]): auto =
  let
    rowA = a.shape[0]
    colA = a.shape[1]
    rowB = b.shape[0]
    colB = b.shape[1]

  when compileOption("boundChecks"):
    if colA != rowB:
      raise newException(IndexError, "Number of columns in the first matrix: " & $(colA) & ", must be the same as the number of rows in the second matrix: " & $(rowB))

  result.data = newSeq[T](rowA * colB)
  result.dimensions = @[colB, rowA]
  result.strides = @[rowA, 1]
  result.offset = addr result.data[0]

  gemm(rowMajor, noTranspose, noTranspose, rowA, colB, rowB, 1, a.offset, colA, b.offset, colB, 0, result.offset, colB)

proc `*`*[B, T](a, b: Tensor[B, T]): Tensor[B, T] {.inline, noSideEffect.} =
  if (a.rank == 2 and b.rank == 2): matmul(a, b, result)
  else: raise newException(ValueError, "Tensor multiplications, not implemented for ranks other thank 2")
    
    
