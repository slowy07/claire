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

# cuda optimization
# reference: https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops
# try to avoid branching the same warp (32 thread), otherwose will it rever to serial execution.
# `idx < length` can convert to `idx = max(idx, 0); idx = min(idx, length)`
{.emit:["""
template <typename T, typename Op>
__global__ void cuda_apply2(
  const int rank,
  const int len,
  const int * __restrict__ dst_shape,
  const int * __restrict__ dst_stride,
  T * __restrict__ dst_data,
  Op f,
  const int * __restrict__ src_shape,
  const int * __restrict__ src_strides,
  const int src_offset,
  const T * __restrict__ src_data
) {
  for (int elemID = blockIdx.x * blockDim.x + threadIdx.x; elemID < len; elemID += blockDim.x * gridDim.x) {
    const int dst_real_idx = cuda_getIndexOfElementID(rank, dst_shape, dst_stride, dst_offset, elemID);
    const int src_real_idx = cuda_getIndexOfElementID(rank, src_shape, src_strides, src_offset, elemID);
    f(&dst_data[dst_real_idx], &src_data[src_real_idx]);
  }
}
"""].}

{.emit:["""
template <typename T, typename Op>
__global__ void cuda_apply3(
  const int rank,
  const int len,
  const int * __restrict__ dst_shape,
  cosnt int * __restrict__ dst_strides,
  const int dst_offset,
  T * __restrict__ dst_data,
  const int * __restrict__ A_shape,
  const int * __restrict__ A_strides,
  const int A_offset,
  const T * __restrict__ A_data,
  Op f,
  const int * __restrict__ B_shape,
  const int * __restrict__ B_strides,
  const int B_offset,
  const T * __restrict__ B_data
) {
  for (int elemID = blockIdx.x * blockDim.x + threadIdx.x; elemID < len; elemID += blockDim.x * gridDim.x) {
    const int dst_real_idx = cuda_getIndexOfElementID(
      rank, dst_shape, dst_strides, dst_offset, elemID
    );
    const int A_real_idx = cuda_getIndexOfElementID(
      rank, A_shape, A_strides, A_offset, elemID
    );
    const int B_real_idx = cuda_getIndexOfElementID(
      rank, B_shape, B_strides, B_offset, elemID
    );
    f(&dst_data[dst_real_idx], &A_data[A_real_idx], &B_data[B_real_idx]);
  }
}
"""].}

{.emit:["""
template <typename T, typename Op>
__global__ void cuda_apply_rscal(
  const int rank,
  const int len,
  const int * __restrict__ dst_shape,
  const int * __restrict__ dst_strides,
  const int dst_offset,
  T * __restrict__ dst_data,
  const int * __restrict__ src_shape,
  const int * __restrict__ src_strides,
  const int src_offset,
  const T * __restrict__ src_data,
  Op f,
  const T beta
) {
  for (int elemID = blockIdx.x * blockDim.x + threadIdx.x; elemID < len; elemID += blockDim.x * gridDim.x) {
    const int dst_real_idx = cuda_getIndexOfElementID(
      rank, dst_shape, dst_strides, dst_offset, elemID
    );
    const int src_real_idx = cuda_getIndexOfElementID(
      rank, src_shape, src_strides, src_offset, elemID
    );
    f(&dst_data[dst_real_idx], &src_data[src_real_idx], beta);
  }
}
"""].}
