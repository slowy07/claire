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

proc gemm_macro_kernel[T](mc, nc, kc: int, alpha: T, beta: T, C: var seq[T], offC: int, incRowC, incColC: int, buffer_A: var ref array[MCKC, T], buffer_B: var ref array[KCNC, T], buffer_C: var ref array[MRNR, T]) { .noSideEffect. } =
  let mp = (mc + MR - 1) div MR
  let np = (nc + NR - 1) div NR
  
  let mod_mr = mc mod MR
  let mod_nr = nc mod NR

  var mr: int
  var nr: int

  for j in 0 .. < np:
    nr = if (j != np - 1 or mod_nr == 0): NR
         else: mod_nr
    for i in 0 .. < mp:
      mr = if (i != mp - 1 or mod_mr == 0): MR
           else: mod_mr
      if (mr == MR and nr == NR):
        gemm_micro_kernel(kc, alpha, buffer_A, i * KC * MR, buffer_B, j * kc * NR, beta, C, i * MR * incRowC + j * NR * incColC + offC, incRowC, incColC)
      else:
        gemm_micro_kernel(kc, alpha, buffer_A, i * kc * MR, buffer_B,j * kc * NR, 0.T, buffer_C, 0, 1, MR)
        gescal(mr, nr, beta, C, i * MR * incRowC + j * NR * incColC + offC, incRowC, incColC)
        geaxpy(mr, nr, 1.T, buffer_C, 1, MR, C, i * MR * incRowC + j * NR * incColC + offC, incRowC, incColC
