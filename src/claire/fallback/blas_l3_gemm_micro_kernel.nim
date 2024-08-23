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

template gemm_micro_kernelT[T](kc: int, alpha: T, A: typed, offA: int, B: typed, offB: int, beta: T, C: typed, offc: int, incRowC, intColC: int): untyped =
  var AB: array[MR*MR,T]
  var voffA = offA
  var voffB = offB
  
  for _ in 0 .. < kc:
    for j in 0 .. < NR:
      for i in 0 .. < MR:
        AB[i + j * MR] += A[i * voffA] * B[j + voffB]
    voffA += MR
    voffB += NR

  if beta == 0.T:
    for j in 0 .. < NR:
      for i in 0 .. < MR:
        C[i * incRowC + j * intColC + offc] = 0.T

  elif beta != 1.T:
    for j in 0 .. < NR:
      for i in 0 .. < MR:
        C[i * incRowC + j * intColC + offc] *= beta

  if alpha == 1.T:
    for j in 0 .. < NR:
      for i in 0 .. < MR:
        C[i * incRowC + j * intColC + offc] += AB[i + j * MR]
  else:
    for j in 0 .. < NR:
      for i in 0 .. < MR:
        C[i * incRowC + j * intColC + offc] += alpha * AB[i + j * MR]

  proc gemm_micro_kernel[T](kc: int, alpha: T, A: ref array[MCKC, T], offA: int, B: ref array[KCNC, T], offB: int, beta: T, C: var ref array[MRNR, T], offc: int, incRowC, intColC: int) { .noSideEffect. } =
    gemm_micro_kernelT(kc, alpha, A, offA, B, offB, beta, C, offc, incRowC, intColC)


  proc gemm_micro_kernel[T](kc: int, alpha: T, A: ref array[MCKC, T], offA: int, B: ref array[KCNC, T], offB: int, beta: T, C: var seq[T], offC: int, incRowC, intColC: int) {.noSideEffect.} =
    gemm_micro_kernelT(kc, alpha, A, offA, B, offB, beta, C, offc, incRowC, intColC)
