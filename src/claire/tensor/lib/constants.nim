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

# this set max number of dimension that claire can support, 8 int64 can be
# found in cache x86-64, therefore, for optimal performance, the meteadata
# should consist of seven items plus one integer for length, deep learning
# requires max rank of six for 3D films [batch, time, depth, height, width,
# color / feature, channels]
const MAXRANK* = 7

const CUDA_HOF_TPB* = 32 * 32
# should be (grid-stride + threadsPerBlock - 1) div threadsPerBlock
# reference: https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
# lower alloe thread re-use and limit everhead of thread creation/destruction
const CUDA_HOF_BPG*: cint = 256

# tensor number of elements threshold before using openmp
const OMP_FOR_THRESHOLD* = 1000
# max number of expected openmp threads
const OMP_MAX_REDUCE_BLOCKS* = 8

# processor optiomization (AVX, AVX2, ARM neon ... if aplicable)
when defined(native):
  {.passC: "-march=native".}
