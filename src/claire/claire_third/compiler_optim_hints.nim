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

const CLAIRE_MEM_ALIGN*{.intdefine.} = 64
static:
  assert CLAIRE_MEM_ALIGN != 0, "Alignment " & $CLAIRE_MEM_ALIGN & "must power of 2"
  assert (CLAIRE_MEM_ALIGN and (CLAIRE_MEM_ALIGN - 1)) == 0, "Alignment " & $CLAIRE_MEM_ALIGN & "must power of 2"

template withCompilerOptimHints*() =
  {.pragma: align_variable, codegenDecl: "$# $# __attribute__((aligned (" & CLAIRE_MEM_ALIGN & ")))".}
  when not defined(vcc):
    {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}
  else:
    {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}

const withBuiltins = defined(gcc) or defined(clang) or defined(icc)
type
  PrefetchRW* {.size: cint.sizeof.} = enum
    Read = 0
    Write = 1
  PrefetchLocality* {.size: cint.sizeof.} = enum
    NoTemporalLocality = 0
    LowTomporalLocality = 1
    ModerateTemporalLocality = 2
    HighTemporalLocality = 3

when withBuiltins:
  proc builtin_assume_aligned(data: pointer, alignment: csize_t): pointer {.importc: "__builtin_assume_aligned", nodecl.}
  proc builtin_prefetch(data: pointer, rw: PrefetchRW, locality: PrefetchLocality) {.importc: "__builtin_prefetch", nodecl.}
  
when defined(cpp):
  proc static_cast[T: ptr](input: pointer): T {.importcpp: "static_cast<'0>(@)".}

template assume_aligned*[T](data: ptr T, alignment: static int = CLAIRE_MEM_ALIGN):
  ptr T =
    when defined(cpp) and withBuiltins:
      static_cast[ptr T](builtin_assume_aligned(data, alignment))
    elif withBuiltins:
      cast[ptr T](builtin_assume_aligned(data, alignment))
    else:
      data

template prefetch*[T](data: ptr(T or UncheckedArray[T]), rw: static PrefetchRW = Read, locality: static PrefetchLocality = HighTemporalLocality) =
  when withBuiltins:
    builtin_prefetch(data, rw, locality)
  else:
    discard

template pragma_ivdep() {.used.} =
  when defined(gcc):
    {.emit: "#pragma GCC ivdep".}
  else:
    {.emit: "pragma ivdep".}

template withCompilerFunctionHints() {.used.} =
  {.pragma: aligned_ptr_result, codegenDecl: "__attribute__((assumed_aligned(" & $CLAIRE_MEM_ALIGN & "))) $# $#$#".}
  {.pragma: malloc, codegenDecl: "__attribute__((malloc)) $# $#$#".}
  {.pragma: simd, codegenDecl: "__attribute__((simd)) $# $#$#".}
  {.pragma: hot, codegenDecl: "__attribute__((hot)) $# $#$#".}
  {.pragma: cold, codegenDecl: "__attribute__((cold)) $# $#$#".}
  {.pragma: gcc_pure, codegenDecl: "__attribute__((pure)) $# $#$#".}
  {.pragma: gcc_const, codegenDecl: "__attribute__((const)) $# $#$#".}

