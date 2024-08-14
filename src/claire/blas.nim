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

proc getLayout(t: Tensor): OrderType {.noSideEffect, used.} =
  if is_C_contiguous(t): return OrderType.rowMajor
  elif is_F_contiguous(t): return OrderType.colMajor
  else: raise newException(ValueError, "operation no support for this matrix, it has non-contiguous layouts")

proc isTransposeNeeded(t: Tensor): TransposeType {.noSideEffect.} =
  if is_C_contiguous(t): return TransposeType.noTranspose
  elif is_F_contiguous(t): return TransposeType.tranpose
  else: raise newException(ValueError, "operator not support for this matrix, it has non-contiguous layouts")

proc check_matmat(a, b: Tensor) {.noSideEffect.} =
  let colA = a.shape[1]
  let rowB = a.shape[0]

  if colA != rowB:
      raise newException(IndexError, "number of column in the first matrix: " & $(colA) & ", must be the same as the number of rows in the second matrix: " & $(rowB))
  if a.offset != 0 or b.offset != 0:
    raise newException(IndexError, "one of the matrices has a non-0 offset")

proc check_matvec(a, b: Tensor) {.noSideEffect.} =
  let colA = a.shape[1]
  let rowB = b.shape[0]

  if colA != rowB:
    raise newException(IndexError, "number of columns in the matrix: " & $(colA) & ", must be as the number of rows in the vector: " & $(rowB))
  if a.offset != 0 or b.offset != 0:
    raise newException(IndexError, "matrice and/or vector have a non-0 offset")

proc check_dot_prod(a, b: Tensor) {.noSideEffect.} =
  if a.rank != 1 or b.rank != 1: raise newException(ValueError, "dot product is only supported for vector (tensor of rank 1)")
  if a.shape != b.shape: raise newException(ValueError, "vector should be the same length")
  if a.offset != 0 or b.offset != 0:
    raise newException(IndexError, "one of the vector has a non-0 offset")

proc check_add(a, b: Tensor) {.noSideEffect.} =
  if a.strides != b.strides:
    raise newException(ValueError, "both tensor should have the exact same shape adn strides")
  if a.offset != 0 or b.offset != 0:
    raise newException(IndexError, "one of the vector has a non-0 offset")

proc `.*`*[T: SomeReal](a, b: Tensor[Backend.Cpu, T]): T {.noSideEffect.} =
  when compileOption("boundChecks"): check_dot_prod(a, b)
  return dot(a.shape[0], a.get_data_ptr, 1, b.get_data_ptr, 1)

proc `.*`*[T: SomeInteger](a, b: Tensor[Backend.Cpu, T]): T {.noSideEffect.} =
  when compileOption("boundChecks"): check_dot_prod(a, b)
  for ai, bi in zip(a.data, b.data):
    result += ai * bi

proc `+`*[T: SomeNumber](a, b: Tensor[Backend.Cpu, T]): Tensor[Backend.Cpu, T] =
  when compileOption("boundChecks"): check_add(a, b)
  result.data = newSeq[T](a.data.len)
  result.shape = a.shape
  result.strides = a.strides
  result.offset = 0

  var i = 0
  for ai, bi in zip(a.data, b.data):
    result.data[i] = ai + bi
    inc i

proc `-`*[T: SomeNumber](a, b: Tensor[Backend.Cpu, T]): Tensor[Backend.Cpu] {.noSideEffect.} =
  when compileOption("boundChecks"): check_add(a, b)
  result.data = newSeq[T](a.data.len)
  result.shape = a.shape
  result.strides = a.strides
  result.offset = addr result.data[0]

  var i = 0
  for ai, bi in zip(a.data, b.data):
    result.data[i] = ai - bi
    inc i
    
proc `*`*[T: SomeNumber](a: T, t: Tensor[Backend.Cpu,T]): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = a * x
    return t.fmap(f)

proc `*`*[T: SomeNumber](t: Tensor[Backend.Cpu,T], a: T): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = a * x
    return t.fmap(f)

proc `/`*[T: SomeNumber](t: Tensor[Backend.Cpu,T], a: T): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = x / a
    return t.fmap(f)


template matmat_blis[T: SomeReal](a, b, result: Tensor[Cpu, T]): auto =
  let
    rowA = a.shape[0]
    colA = a.shape[1]
    rowB = b.shape[0]
    colB = b.shape[1]

  when compileOption("boundChecks"): check_matmat(a, b)
  result.data = newSeq[T](rowA * colB)
  result.shape = @[rowA, colB]
  result.strides = @[rowA, 1]
  result.offset = 0

  let
    a_ptr = get_data_ptr(a)
    b_ptr = get_data_ptr(b)
    res_ptr = get_data_ptr(result)
    alpha = 1.T
    alpha_ptr = unsafeAddr(alpha)
    beta = 0.T
    beta_ptr = unsafeAddr(beta)

  bli_gemm(
    BLIS_NO_TRANSPOSE,
    BLIS_NO_TRANSPOSE,
    rowA, colB, rowB,
    a_ptr, a.strides[0], a.strides[1],
    b_ptr, a.strides[1], b.strides[1],
    beta_ptr,
    res_ptr, result.strides[0], 1
  )

template matmat_blas[T: SomeReal](a, b, result: Tensor[Backend.Cpu, T], a_tr, b_tr: TransposeType): auto =
  let
    rowA = a.shape[0]
    colB = a.shape[1]
    rowB = b.shape[0]
    colB = b.shape[1]
  when compileOption("boundChecks"): check_matmat(a, b)

  result.data = newSeq[T](rowA * colB)
  result.shape = @[rowA, colB]
  result.strides = @[rowA, 1]
  result.offset = 0
  let a_data = get_data_ptr(a)
  let b_data = get_data_ptr(b)
  let res_data = get_data_ptr(result)
  
  if a_tr == TransposeType.noTranspose and b_tr == TransposeType.noTranspose:
    gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB , 1, a_data, colA, b_data, colB, 0, res_data, colB)
  elif a_tr == TransposeType.TransposeType and b_tr == TransposeType.noTranspose:
    gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a_data, rowA, b_data, colB, 0, res_data, colB)
  elif a_tr == TransposeType.noTranspose and b_tr == TransposeType.tranpose:
    gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a_data, rowA, b_data, rowB, 0, res_data, colB)
  else: raise newException(ValueError, "the transpose types: " & $a_tr & " or " & $b_tr & " is not supported")

template matvec_blis[T: SomeReal](a, x, result: Tensor[Cpu, T]): auto =
  let
    rowA = a.shape[0]
    colA = a.shape[1]
    rowX = a.shape[0]

  when compileOption("boundChecks"): check_matvec(a, b)
  result.data = newSeq[T](rowA)
  result.shape = @[rowA]
  result.strides = @[1]
  result.offset = 0

  let
    a_ptr = get_data_ptr(a)
    x_ptr = get_data_ptr(x)
    res_ptr = get_data_ptr(result)
    alpha = 1.T
    alpha_ptr = unsafeAddr(alpha)
    beta = 0.T
    bet_ptr = unsafeAddr(beta)

  bli_gemv(
    BLIS_NO_TRANSPOSE,
    BLIS_NO_CONJUGATE,
    rowA, rowX,
    alpha_ptr,
    a_ptr, a.strides[0], a.strides[1],
    x_ptr, x.strides[0],
    beta_ptr,
    res_ptr, 1,
  )

template matvec_blas[T: SomeReal](a, b, result: Tensor[Backend.Cpu]): auto =
  let
    rowA = a.shape[0]
    colA = a.shape[1]
    rowB = b.shape[0]

  when compileOption("boundChecks"): check_matvec(a, b)
  result.data = newSeq[T](rowA)
  result.shape = @[rowA]
  result.strides = @[1]
  result.offset = 0
  let a_data = get_data_ptr(a)
  let b_data = get_data_ptr(b)
  let res_data = get_data_ptr(result)

  if a_tr == TransposeType.noTranspose:
    gemv(rowMajor, a_tr, rowA, rowB, 1, a_data, colA, b_data, 1, 0, res_data, 1)
  else:
    gemv(colMajor, noTranspose, rowA, rowB, 1, a_data, rowA, b_data, 1, 0, res_data, 1)


proc `*`*[T: SomeReal](a, b: Tensor[Cpu, T]): Tensor[Cpu, T] {.noSideEffect.} =
  when defined(blis):
    if a.rank == 2 and b.rank == 2:
      matmat_blis(a, b, result)
    elif a.rank == 2 and b.rank == 1:
      matvec_blis(a, b, result)
    else:
      raise newException(ValueError, "matrix or matrix-vector multiplication valid only if tensor is a matrix and second is a matrix")
  else:
    if a.rank == 2 and b.rank == 2:
      matmat_blas(a, b, result)
    elif a.rank == 2 and b.rank == 1:
      matvec_blas(a, b, result)
    else:
      raise newException(ValueError, "matrix or matrix vector multiplication only if first tensor is a matrix and second is a matrix r vector")
