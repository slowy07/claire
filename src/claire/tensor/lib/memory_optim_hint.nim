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

when (NimMajor, NimMinor) < (1, 4):
  import ../../std_ver_types

template withMemoryOptimHints*() =
  when not defined(js):
    {.pragma: align64, codegenDecl: "$# $# __attribute__((aligned(64)))".}
    {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}
  else:
    {.pragma: align64.}
    {.pragma: restrict.}

const withBuiltins = defined(gcc) or defined(clang)
when withBuiltins:
  proc builtin_assume_aligned[T](data: ptr T, n: csize_t): ptr T {.importc: "__builtin_assume_aligned", nodecl.}

when defined(cpp):
  proc static_cast[T](input: T): T
    {.importcpp: "static_cast<'0>(@)".}

template assume_aligned*[T](data: ptr T, n: csize_t): ptr T =
  when defined(cpp) and withBuiltins:
    static_cast builtin_assume_aligned(data, n)
  elif withBuiltins:
    builtin_assume_aligned(data, n)
  else:
    data
