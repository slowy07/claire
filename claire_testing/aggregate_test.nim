import ../claire
import unittest, math

suite "testing aggregation functions":
  let t = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11]
  ].toTensor(Cpu)

  test "sum all elements":
    check: t.sum == 66

  test "sum over axis":
    let row_sum = [[18, 22, 26]].toTensor(Cpu)
    let col_sum = [
      [3],
      [12],
      [21],
      [30]
    ].toTensor(Cpu)
    check: t.sum(axis=0) == row_sum
    check: t.sum(axis=1) == col_sum

  test "mean of all elements":
    check: t.astype(float).mean == 5.5

  test "mean over axis":
    let row_mean = [[4.5, 5.5, 6.5]].toTensor(Cpu)
    let col_mean = [
      [1.0],
      [4.0],
      [7.0],
      [10.0]
    ].toTensor(Cpu)
    check: t.astype(float).mean(axis=0) == row_mean
    check: t.astype(float).mean(axis=1) == col_meana

  test "generic aggregate function":
    proc addition[T](a, b: T): T =
      return a + b
    proc addition_inplace[T](a: var T, b: T) =
      a += b

    check: t.agg(addition, start_val = 0) == 66
    var z = 0
    z.agg_inplace(addition_inplace, t)
    check: z == 66

    let row_sum = [[18, 22, 26]].toTensor(Cpu)
    let col_sum = [
      [3],
      [12],
      [21],
      [30]
    ].toTensor(Cpu)

    var z1 = zeros([1, 3], int, Cpu)
    var z2 = zeros([4, 1], int, Cpu)

    check: t.agg(`+`, axis=0, start_val = z1) == row_sum
    check: t.agg(`+`, axis=1, start_val = z2) == col_sum
    z1.agg_inplace(`+=`, t, axis=0)
    z2.agg_inplace(`+=`, t, axis=1)

    check: z1 == row_sum
    check: z2 == col_sum
