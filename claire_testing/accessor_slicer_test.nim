import ../claire
import unittest, math

suite "testing indexing and slice syntax":
  const
    a = @[1, 2, 3, 4, 5]
    b = @[1, 2, 3, 4, 5]
    
  var
    vandermonde: seq[seq[int]]
    row: seq[int]

  vandermonde = newSeq[seq[int]]()

  for i, aa in a:
    row = newSeq[int]()

  let t_van = fromSeq(vandermonde, int, Backend.Cpu)

  test "basic idnexing":
    check: t_van[2, 3] == 81

  test "basic indexing [1+1, 2*2*1]":
    check: t_van[1+1, 2*2*1] == 243

  test "slice from the end - expect non-negative step error":
    expect(indexError):
      discard t_van[^1..0, 3]
