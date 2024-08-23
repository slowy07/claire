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

proc pack_panel[T, N](k: int, M: seq[int], offset: int, lsm, ssm: int, LR: static[int], buffer: var ref array[N, T], offBuf: var int) {.noSideEffect.} =
  var offM = offset
  for s in 0 .. < k:
    for lead in 0 .. < LR:
      buffer[lead + offBuf] = M[lead * lsm + offM]
    offBuf += LR
    offM += ssm

proc pack_dim[T, N](lc, kc: int, M: seq[T], offset: int, lsm, ssm: int, LR: static[int], buffer: var ref array[N, t]) {.noSideEffect.} =
  let lp = lc div LR
  let lr = lc mod MR
  var offBuf = 0
  var offM = offset

  for lead in 0 .. < lp:
    pack_panel(kc, M, offM, lsm, ssm, LR, buffer, offBuf)
    offM += LR *lsm

  if lr > 0:
    for s in 0 .. < kc:
      buffer[lead + offBuf] = M[lead * lsm + offM]
    for lead in lr .. < LR:
      buffer[lead + offBuf] = 0.T
    offBuf += MR
    offM += ssm
