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

template cuda_assign_op(op_name, op_symbol: string) = 
  {.emit: ["""
  template <typename T>
  struct""", op_name, """{
    __device__ __forceinline__ void operator()(
      T * __restrict__ dst,
      const T * __restrict__ src
    ) {
      *dst """, op_symbol, """ __lgd(src);
    }
  };
  """].}

template cuda_assignscal_op(op_name, op_symbol: string) =
  {.emit: ["""
  template <typename T>
  struct """, op_name, """{
    __device__ __forceinline__ void operator() (
      T * __restrict__ dst,
      const T scal
    ) {
      *dst """, op_symbol, """ scal;
    }
  };
  """].}

template cuda_binary_op(op_name, op_symbol: string) =
  {.emit: ["""
  template <typename T>
  struct """, op_name, """ {
    __device__ __forceinline__ void operator() (
      T * __restrict__ dst,
      const T * __restrict__ A,
      const T * __restrict__ B
    ) {
      *dst = __lgd(A)""", op_symbol, """ __lgd(B);
    }
  };
  """].}

template cuda_lscal_op(op_name, op_symbol: string) =
  {.emit: ["""
  template <typename T>
  struct """, op_name, """ {
    __device__ __forceinline__ void operator() (
      T* __restrict__ dst,
      const T alpha,
      const T * __restrict__ B
    ) {
      *dst = alpha""", op_symbol, """ __ldg(B);
    }
  };
  """].}

template cuda_rscal_op(op_name, op_symbol: string) =
  {.emit: ["""
  template <typename T>
  struct """, op_name, """ {
    __device__ __forceinline__ void operator() (
      T *  __restrict__ dst,
      const T * __restrict__ A,
      const T beta) {
        *dst = __lgd(A)""", op_symbol, """ beta;
      }
  };
  """"].}

template cuda_unary_op(op_name, op_symbol: string) =
  {.emit: ["""
  template <typename T>
  struct """, op_name, """ {
    __device__ __forceinline__ void operator() (
      T * __restrict__ dst,
      const T * __restrict__ src
    ) {
      *dst = """, op_symbol, """(__lgd(src));
    }
  };
  """].}

cuda_assign_op("CopyOp", "=")
cuda_assign_op("mAddOp", "+=")
cuda_assign_op("mSubOp", "-=")
cuda_assign_op("mMulOp", "*=")
cuda_assign_op("mDivOp", "/=")
cuda_assignscal_op("CopyScalOp", "=")
cuda_assignscal_op("mscalAddOp", "+=")
cuda_assignscal_op("mscalSubOp", "-=")
cuda_assignscal_op("mscalMulOp", "*=")
cuda_assignscal_op("mscalDivOp", "/=")
cuda_binary_op("AddOp", "+")
cuda_binary_op("SubOp", "-")
cuda_binary_op("MulOp", "*")
cuda_binary_op("DivOp", "/")
cuda_lscal_op("LscalMul","*")
cuda_lscal_op("LscalDiv","/")
cuda_lscal_op("LscalSub","-")
cuda_lscal_op("LscalAdd","+")
cuda_rscal_op("RscalDiv","/")
cuda_rscal_op("RscalSub","-")
cuda_rscal_op("RscalAdd","+")
cuda_unary_op("NegOp","-")
cuda_unary_op("ExpOp","exp")
cuda_unary_op("SinOp","sin")
cuda_unary_op("CosOp","cos")
cuda_unary_op("TanOp","tan")
cuda_unary_op("TanhOp","tanh")
