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

proc geaxpy[T](m, n: int, alpha: T, X: ref array[MRNR, T], incRowX, incColX: int, Y: var seq[T], offY: int, incRowY, incColY: int) {.noSideEffect.} =
  if alpha != 1.T:
    for j in 0 .. < n:
      for i in 0 .. < m:
        Y[i * incRowY + j * incColY + offY] += alpha * X[i * incRowX + j *incColX]
  else:
    for j in 0 .. < n:
      for i in 0 .. < m:
        Y[i * incRowY + j * incColY + offY] += X[i * incRowX + j * incColX]

proc gescal[T](m, n: int, alpha: T, X: var seq[T], offX: int, incRowX, incColX: int) {.noSideEffect.} =
  if alpha != 0.T:
    for j in 0 .. < n:
      for i in 0 .. < m:
        X[i * incRowX + j * incColX + offX] *= alpha
  else:
    for j in 0 .. < n:
      for i in 0 .. < m:
        X[i * incRowX + j * incColX + offX] = 0
