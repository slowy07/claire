import ../claire
import unittest

suite "Basic Linear Algebra subprogram":
  test "General matrix to matrix multiplication":
    let a = @[@[1.0, 2, 3], @[4.0, 5, 6]]
    let b = @[@[7.0, 8], @[9.0, 10], @[11.0, 12]]

    let ta = fromSeq(a, float64, Backend.Cpu)
    let tb = fromSeq(b, float64, Backend.Cpu)

    let expected = @[@[58.0, 64], @[139.0, 154]]
    let t_expected = fromSeq(expected, float64, Backend.Cpu)

    check: ta * tb == t_expected
