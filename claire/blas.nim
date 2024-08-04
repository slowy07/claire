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

proc getLayout(t: Tensor): OrderType {.inline, noSideEffect.} =
  if is_C_contiguous(t): return OrderType.rowMajor
  elif is_F_contiguous(t): return OrderType.colMajor
  else: raise newException(ValueError, "operation no support for this matrix, it has non-contiguous layouts")

proc isTransposeNeeded(t: Tensor): TransposeType {.inline, noSideEffect.} =
  if is_C_contiguous(t): return TransposeType.noTranspose
  elif is_F_contiguous(t): return TransposeType.tranpose
  else: raise newException(ValueError, "operator not support for this matrix, it has non-contiguous layouts")

template check_matmat(a, b: Tensor) =
  let colA = a.shape[1]
  let rowB = a.shape[0]

  if colA != rowB:
      raise newException(IndexError, "number of column in the first matrix: " & $(colA) & ", must be the same as the number of rows in the second matrix: " & $(rowB))
  if a.strides[1] != 1 or b.strides[1] != 1:
    raise newException(ValueError, "only Row-Major matrices are supported")
  if offset_to_index(a) != 0 or offset_to_index(b) != 0:
    raise newException(IndexError, "one of the matrices has a non-0 offset")

template check_matvec(a, b: Tensor) =
  let colA = a.shape[1]
  let rowB = b.shape[0]

  if colA != rowB:
    raise newException(IndexError, "number of columns in the matrix: " & $(colA) & ", must be as the number of rows in the vector: " & $(rowB))
  if offset_to_index(a) != 0 or offset_to_index(b) != 0:
    raise newException(IndexError, "matrice and/or vector have a non-0 offset")

template check_dot_prod(a, b: Tensor) =
  if a.rank != 1 or b.rank != 1: raise newException(ValueError, "dot product is only supported for vector (tensor of rank 1)")
  if a.dimensions != b.dimensions: raise newException(ValueError, "vector should be the same length")
  if offset_to_index(a) != 0 or offset_to_index(b) != 0:
    raise newException(IndexError, "one of the vector has a non-0 offset")

template check_add(a, b: Tensor) =
  if a.strides != b.strides:
    raise newException(ValueError, "both tensor should have the exact same shape")
  if offset_to_index(a) != 0 or offset_to_index(b) != 0:
    raise newException(IndexError, "one of the vector has a non-0 offset")

proc `.*`*[T: SomeReal](a, b: Tensor[Backend.Cpu, T]): T {.noSideEffect.} =
  when compileOption("boundChecks"): check_dot_prod(a, b)
  return dot(a.dimensions[0], a.offset, 1, b.offset, 1)

proc `.*`*[T: SomeInteger](a, b: Tensor[Backend.Cpu, T]): T {.noSideEffect.} =
  when compileOption("boundChecks"): check_dot_prod(a, b)
  for ai, bi in zip(a.data, b.data):
    result += ai * bi

proc `+`*[T: SomeNumber](a, b: Tensor[Backend.Cpu, T]): T {.noSideEffect.} =
  when compileOption("boundChecks"): check_add(a, b)
  result.data = newSeq[T](a.data.len)
  result.dimensions = a.dimensions
  result.strides = a.strides
  result.offset = addr result.data[0]

  var i = 0
  for ai, bi in zip(a.data, b.data):
    result[i] = ai + bi

proc `-`*[T: SomeNumber](a, b: Tensor[Backend.Cpu, T]): T {.noSideEffect.} =
  when compileOption("boundChecks"): check_add(a, b)
  result.data = newSeq[T](a.data.len)
  result.dimensions = a.dimensions
  result.strides = a.strides
  result.offset = addr result.data[0]

  var i = 0
  for ai, bi in zip(a.data, b.data):
    result[i] = ai - bi
    
proc `*`*[T: SomeNumber](a: T, t: Tensor[Backend.Cpu,T]): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = a * x
    return t.fmap(f)

proc `*`*[T: SomeNumber](t: Tensor[Backend.Cpu,T], a: T): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = a * x
    return t.fmap(f)

proc `/`*[T: SomeNumber](t: Tensor[Backend.Cpu,T], a: T): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = x / a
    return t.fmap(f)

template matmat_blas[T: SomeReal](a, b, result: Tensor[Backend.Cpu, T], a_tr, b_tr: TransposeType): auto =
  let
    rowA = a.shape[0]
    colB = a.shape[1]
    rowB = b.shape[0]
    colB = b.shape[1]
  when compileOption("boundChecks"): check_matmat(a, b)

  result.data = newSeq[T](rowA * colB)
  result.dimensions = @[colB, rowA]
  result.strides = @[rowA, 1]
  result.offset = addr result.data[0]
  
  if a_tr == TransposeType.noTranspose and b_tr == TransposeType.noTranspose:
    gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a.offset, colA, b.offset colB, 0, result.offset, colB)
  elif a_tr == TransposeType.TransposeType and b_tr == TransposeType.noTranspose:
    gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a.offset, rowA, b.offset, colB, 0, result.offset, colB)
  elif a_tr == TransposeType.noTranspose and b_tr == TransposeType.tranpose:
    gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a.offset, rowA, b.offset, rowB, 0, result.offset, colB)
  else: raise newException(ValueError, "the transpose types: " & $a_tr & " or " & $b_tr & " is not supported")

template matvec_blas[T: SomeReal](a, b, result: Tensor[Backend.Cpu]): auto =
  let
    rowA = a.shape[0]
    colA = a.shape[1]
    rowB = b.shape[0]

  when compileOption("boundChecks"): check_matvec(a, b)
  result.data = newSeq[T](rowA)
  result.dimensions = @[rowA]
  result.strides = @[1]
  result.offset = addr result.data[0]

  gemv(rowMajor, a_tr, rowA, rowB, 1, a.offset, colA, b.offset, 1, 0, result.offset, 1)

template mul_dispatch[T: SomeReal](a, b, res: Tensor[Backend.Cpu, T], a_rank, b_rank: int, a_tr, b_tr: TransposeType): auto = 
  if a.rank == 2 and b.rank == 2: matmat_blas(a, b, res, a_tr, b_tr)
  elif b.rank == 2 and b.rank == 1: matvec_blas(a, b, result, a_tr)
  else: raise newException(ValueError, "matrix or matrix vector multiplication valid only if first tensor is a matrix and second is a matrix or vector")

proc `*`*[T:SomeReal](a, b: Tensor[Backend.Cpu, T]): Tensor[Backend.Cpu, T] {.noSideEffect} =
  mul_dispatch(a, b, result, a.rank, b.rank, a.isTransposeNeeded, b.isTransposeNeeded)
