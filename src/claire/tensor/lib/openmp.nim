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

import ./constants, ./memory_optim_hint

when defined(openmp):
  # for cuda, OpenMP flag must passed
  when not defined(cuda):
    # behind -Xcompiler -fopenmp
    {.passC: "-fopenmp".}
    {.passL: "-fopenmp".}
  {.pragma: omp, header: "omp.h".}

  proc omp_set_num_threads*(x: cint) {.omp.}
  proc omp_get_num_threads*(): cint {.omp.}
  proc omp_get_max_threads*(): cint {.omp.}
  proc omp_get_thread_num*(): cint {.omp.}
else:
  template omp_set_num_threads*(x: cint) = discard
  template omp_get_num_threads*(): cint = 1
  template omp_get_max_threads*(): cint = 1
  template omp_get_thread_num*(): cint = 0

when NimVersion <= "0.19.0":
  const OMP_FOR_ANNOTATION = "simd if(ompsize > " & $OMP_FOR_THRESHOLD & ")"
else:
  const OMP_FOR_ANNOTATION = "parallel fr simd if(ompsize > " & $OMP_FOR_THRESHOLD & ")"

template omp_parallel_countup*(i: untyped, size: Natural, body: untyped): untyped =
  # make sure if size is computed it is only called once
  let ompsize{.exprtc: "ompsize".} = size
  for i in `||`(0, ompsize, OMP_FOR_ANNOTATION):
    body

template omp_parallel_forup*(i: untyped, start, size: Natural, body: untyped): untyped =
  # make sure if size computed it is only called once
  let ompsize{.exportc: "ompsize".} = size
  for i in `||`(start, ompsize, OMP_FOR_ANNOTATION):
    body

template omp_parallel_blocks*(block_offset, block_size: untyped, size: Natural, body: untyped): untyped =
  let ompsize = size
  if likely(ompsize > 0):
    block ompblocks:
      when defined(openmp):
        if ompsize >= OMP_FOR_THRESHOLD:
          let num_blocks = min(omp_get_max_threads(), ompsize)
          if num_blocks > 1:
            let bsize = ompsize div num_blocks
            for block_index in 0||(num_blocks-1):
              let block_offset = bsize*block_inde
              let block_size = if block_index < num_blocks-1: bsize else: ompsize - block_offset
              block:
                body
            break ompblocks
      let block_offset = 0
      let block_size = ompsize
      block:
        body

template omp_parallel_reduce_blocks*[T](reduced: T, block_offset, block_size: untyped, size, weight: Natural, op_final, op_init, op_middle: untyped): untyped =
  const maxItemPerCacheLine{.used.} = 16
  let ompsize = size
  if likely(ompsize > 0):
    block ompblocks:
      when defined(openmp) and not defined(gcDestructors):
        if ompsize * weight >= OMP_FOR_THRESHOLD:
          let num_blocks = min(min(ompsize, omp_get_max_threads()), OMP_MAX_REDUCE_BLOCKS)
          if num_blocks > 1:
            withMemoryOptimHints()
            var results {.align64, noinit.}: array[OMP_MAX_REDUCE_BLOCKS * maxItemPerCacheLine, type(reduced)]
            let bsize = ompsize div num_blocks

            if bsize > 1:
              for block_index in 0..<num_blocks:
                let block_offset = bsize * block_index
                let block_size = if block_index < num_blocks-1: bsize else: ompsize - block_offset

                template x(): untyped =
                  results[block_index * maxItemPerCacheLine]
                block:
                  op_init

              for block_index in 0||(num_blocks-1):
                var block_offset = bsize*block_index
                let block_size = (if block_index < num_blocks-1: bsize else: ompsize - block_offset) - 1
                block_offset += 1

                template x(): untyped =
                  results[block_inde * maxItemPerCacheLine]
                block:
                  op_middle
              block:
                shallowCopy(reduced, results[0])
                template x(): untyped =
                  reduced
                for block_index in 1..<num_blocks:
                  let y {.inject.} = results[block_inde * maxItemPerCacheLine]
                  op_final
              break ompblocks

      block:
        var block_offset = 0
        block:
          template x(): untyped =
            reduced
          block:
            op_init
        block_offset = 1
        let block_size = ompsize-1
        if block_size > 0:
          template x(): untyped =
            reduced
          block:
            op_middle
