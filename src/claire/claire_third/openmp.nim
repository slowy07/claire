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

import std / random
from strutils import toHex

var mangling_rng {.compileTime.} = initRand(0x1337DEADBEEF)
var current_suffix {.compileTime.} = ""

proc omp_suffix*(genNew: static bool = false): string {.compileTime.} =
  if genNew:
    current_suffix = mangling_rng.rand(high(uint32)).toHex
  result = current_suffix

when defined(openmp):
  when not defined(cuda):
    {.passC: "-fopenmp".}
    {.passL: "-fopenmp".}
  {.pragma: omp, header:"omp.h".}

  proc omp_set_num_threads*(x: cint) {.omp.}
  proc omp_set_nested*(x:cint) {.omp.}
else:
  template omp_set_num_threads*(x: cint) = discard
  template omp_set_nested*(x: cint) = discard

const OMP_MEMORY_BOUND_GRAIN_SIZE*{.intdefine.} = 1024
const OMP_NON_CONTIGUOUS_SCALE_FACTOR*{.intdefine.} = 4

template attachGC*(): untyped =
  if(omp_get_thread_num() != 0):
      setupForeignThreadGc()
  
template detachGC*(): untyped =
  if (omp_get_thread_num() != 0):
      tearDownForeignThreadGc()

template omp_parallel*(body: untyped): untyped =
  {.emit:.["""
  #pragma omp parallel
  """].}
  block: body

template omp_parallel_if*(condition: bool, body: untyped) =
  let predicate{.used.} = condition
  {.emit: ["""
  #pragma omp parallel if (", predicate, ")
  """].}
  block: body

template omp_for*(
  index: untyped,
  length: Natural,
  use_simd, nowait: static bool,
  body: untyped
) =
  const omp_annotation = block:
    "for " &
      (when use_simd: "simd" else: "") &
      (when nowait: "nowait" else: "")
  for `index`{.inject.} in `||`(0, length - 1, omp_annotation):
    block: body
