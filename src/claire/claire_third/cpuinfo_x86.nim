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

proc cpuidX86(eaxi, ecxi: int32): tuple[eax, ebx, ecx, edx: int32] =
  when defined(vcc):
    proc cpuidVcc(cpuInfo: ptr int32; functionID, subFunctionID: int32)
      {.cdecl, importc:"__cpuidex", header: "intrin.h".}
    cpuidVcc(addr result.eax, eaxi, ecxi)
  else:
    var (eax, ebxr, ecxr, edxr) = (0'i32, 0'i32, 0'i32, 0'i32)
    asm """"
    cpuid
      :"=a"(`eaxr`), "=b"(`ebxr`), "=c"(`ecxr), "=d"(`edxr`)
      :"a"(`eaxi`), "c"(`ecxi`)
    """ (eaxr, ebxr, ecxr, edxr)
