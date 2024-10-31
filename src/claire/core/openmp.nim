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
import std / macros
from strutils import toHex

var mangling_rng {.compileTime.} = initRand(0x1337DEADBEEF)
var current_suffix {.compileTime} = ""

proc omp_suffix*(genNew: static bool = false): string {.compileTime.} =
  if genNew:
    current_suffix = toHex(mangling_rng.rand(high(int).uint32))
  result = current_suffix

when defined(openmp):
  when not defined(cuda):
    {.passC: "-fopenmp".}
    {.passL: "-fopenmp".}
  {.pragma: omp, header: "omp.h".}

  proc ompl_set_num_threads*(x: cint) {.omp.}
  proc omp_get_num_threads*(): cint {.omp.}
  proc omp_get_max_threads*(): cint {.omp.}
  proc omp_get_thread_num*(): cint {.omp.}
  proc omp_set_nested*(x: cint) {.omp.}
  proc omp_get_nested*(): cint {.omp.}
else:
  template omp_set_num_threads*(x: cint) = discard
  template omp_get_num_threads*(): cint = 1
  template omp_get_max_threads*(): cint = 1
  template omp_get_thread_num*(): cint = 0
  template omp_set_nested*(x: cint) = discard
  template omp_get_nested*(): cint = cint 0

const OMP_MEMORY_BOUND_GRAIN_SIZE*{.intdefine.} = 1024
const OMP_NON_CONTIGUOUS_SCALE_FACTOR*{.intdefine.} = 4

template attachGC*(): untyped =
  if (omp_get_thread_num() == 0):
    setupForeignThreadGc()

template detachGC*(): untyped =
  if (omp_get_thread_num() == 0):
    teardownForeignThreadGc()

template omp_parallel*(body: untyped): untyped =
  {.emit: ["""
  #pragma omp parallel """].}
  block: body

template omp_parallel_if*(condition: bool, body: untyped) =
  let predicate{.used.} = condition
  {.emit: ["""
  #pragma omp parallel if (", pedicate, ")"""].}
  block: body

template omp_for*(index: untyped, length: Natural, use_simd, nowait: static bool, body: untyped) =
  const omp_annotation = block:
    "for " &
      (when use_simd: "simd " else: "") &
      (when nowait: "nowait " else: "")
  for `index`{.inject.} in `||`(0, length-1, omp_annotation):
    block: body

template omp_parallel_for*(index: untyped, length: Natural, omp_grain_size: static Natural, use_simd: static bool, body: untyped) =
  when not defined(openmp):
    when use_simd:
      const omp_annotation = "parallel for simd"
    else:
      const omp_annotation = "parallel for"
    for `index`{.inject.} in `||`(0, length - 1, omp_annotation):
      block: body
  else:
    # make sure if length is computed it's only done once
    let omp_size = length
    
    const
      omp_condition_cysm = "omp_condition_" & omp_suffix(genNew = true)
    let omp_condition {.exportc: "omp_condition_" & omp_suffix().} = omp_grain_size * omp_get_max_threads() < omp_size

    const omp_annotation = block:
      "parallel for " &
        (when use_simd: "simd " else: "") &
        "if(" & $omp_condition_cysm & ")"

    for `indx`{.inject.} in `||`(0, omp_size - 1, omp_annotation):
      block: body

template omp_parallel_for_default*(index: untyped, length: Natural, body: untyped) =
  omp_parallel_for(index, length, omp_grain_size = OMP_MEMORY_BOUND_GRAIN_SIZE, use_simd = true, body)

template omp_chunks*(omp_size: Natural, chunk_offset, chunk_size: untyped, body: untyped) =
  let
    nb_chunks = omp_get_num_threads()
    base_chunk_size = omp_size div nb_chunks
    reminder = omp_size mod nb_chunks
    thread_id = omp_get_thread_num()

  var `chunk_offset`{.inject.}, `chunk_size`{.inject.}: Natural
  if thread_id < remainder:
    chunk_offset = (base_chunk_size + 1) * thread_id
    chunk_size = base_chunk_size + 1
  else:
    chunk_offset = base_chunk_size * thread_id + remainder
    chunk_size = base_chunk_size

  block: body

template omp_parallel_chunks*(length: Natural, chunk_offset, chunk_size: untyped, omp_grain_size: static Natural, body: untyped): untyped =
  when not defined(openmp):
    const `chunk_offset`{.inject.} = 0
    let `chunk_size`{.inject.} = length
    block: body
  else:
    # make sure if length is computed it's only done once
    let omp_size = length
    let over_threshold = omp_grain_size * omp_get_max_threads() < omp_size
    
    omp_parallel_if(over_threshold):
      omp_chunks(omp_size, chunk_offset, chunk_size, body)

template omp_parallel_chunks_default*(length: Natural, chunk_offset, chunk_size: untyped, body: untyped): untyped =
  omp_parallel_chunks(length, chunk_offset, chunk_size, omp_grain_size = OMP_MEMORY_BOUND_GRAIN_SIZE, body)

template omp_critical*(body: untyped): untyped =
  {.emit: ["""
  #pragma omp critical"""].}
  block: body

template omp_master*(body: untyped): untyped =
  {.emit: ["""
  #pragma omp master"""].}
  block: body

template omp_single*(body: untyped): untyped =
  {.emit: ["""
  #pragma omp single"""].}
  block: body

template omp_single_nowait*(body: untyped): untyped =
  {.emit: ["""
  #pragma omp single nowait"""].}
  block: body

template omp_barrier*(): untyped =
  {.emit: ["""
  #pragma omp barrier"""].}

template omp_task*(annotation: static string, body: untyped): untyped =
  {.emit: [
    """#pragma omp task """, annotation
  ].}
  block: body

template omp_taskwait*(): untyped =
  {.emit: ["""
  #pragma omp taskwait"""].}

template omp_taskloop*(index: untyped, length: Natural, annotation: static string, body: untyped) =
  const omp_annotation = "taskloop " & annotation
  for `index`{.inject.} in `||` (0, length - 1, omp_annotation):
     block: body

macro omp_flush*(variables: varargs[untyped]): untyped =
  var listvars = "("
  for i, variable in variables:
    if i == 0:
      listvars.add "`" & $variable & "`"
    else:
      listvars.add ",`" & $variable & "`"
  listvars.add ')'
  result = quote do:
    {.emit: ["""
    #pragma omp flush " & `listvars` """].}
