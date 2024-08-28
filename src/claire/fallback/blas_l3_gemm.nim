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

const MC = 96
const KC = 256
const NC = 4096

const MR = 2
const NR = 2

const MCKC = MC * KC
const KCNC = KC * NC
const MRNR = MR * NR

include ./blas_l3_gemm_packing
include ./blas_l3_gemm_aux
include ./blas_l3_gemm_micro_kernel
include ./blas_l3_gemm_macro_kernel

proc newBufferArray[T: SomeNumber](N: static[int], typ: typedesc[T]): ref array[N, T] {.noSideEffect.} =
  new result
  for i in 0 .. < N:
    result[i] = 0.T

proc gemm_nn[T](m, n, k: int alpha: T, A: seq[T], offA: int, incRowA, incColA: int, B: seq[T], offB: int, incRowB, incColB: int, beta: T, C: var seq[T], offC: int, incRowC, incColc: int) {.noSideEffect.} =
  let
    mb = (m + MC - 1) div MC
    nb = (n + NC - 1) div NC
    kb = (k + KC - 1) div KC

    mod_mc = m mod NC
    mod_nc = n mod NC
    mod_kc = k mod KC

  var mc, nc, kc: int
  var tmp_beta: T

  var buffer_A = newBufferArray(MCKC, T)
  var buffer_B = newBufferArray(KCNC, T)
  var buffer_C = newBufferArray(MRNR, T)

  if alpha == 0.T or k == 0:
    gescal(m, n, beta, C, offC, incRowC, incColC)
    return
  
  for j in 0 .. < nb:
    nc = if (j != nb - 1 or mod_nc == 0): NC
         else: mod_nc

    for k in 0 .. < kb:
      kc = if (k != kb - 1 or mod_kc == 0): KC
           else: mod_kc
      tmp_beta = if k == 0: beta
                 else: 1.T

      pack_dim(nc, kc, B, k * KC * incRowB + j * NC * incColB + offB, incColB, incRowB, NR, buffer_B)
      for i in 0 .. < mb:
        mc = if (i != mb - 1 or mod_mc == 0): MC
             else: mod_mc

        pack_dim(mc, kc, A, i * MC * incRowA + k * KC * incColA + offA, incRowA, incColA, MR, buffer_A)
        gemm_macro_kernel(mc, nc, kc, alpha, tmp_beta, C, i * MC * incRowC + j * NC * incColC + offC, incRowC, incColC, buffer_A, buffer_B, buffer_C)
