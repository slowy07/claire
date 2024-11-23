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

import ../compiler_optim_hints

func align_raw_data*(T: typedesc, p: pointer): ptr UncheckedArray[T] =
  static: assert T is KnownSupportsCopyMem
  withCompilerOptimHints()

  let address = cast[uint](p)
  let aligned_ptr{.restrict.} = block:
    let remainder = address and (CLAIRE_MEM_ALIGN - 1)
    if remainder == 0:
      assume_aligned cast[ptr UncheckedArray[T]](address)
    else:
      let offset = CLAIRE_MEM_ALIGN - remainder
      assume_aligned cast[ptr UncheckedArray[T]](address + offset)
  return aligned_ptr

