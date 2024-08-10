import ../claire.nim
import unittest, math

suite "testing tensor compare":
  test "test for [1..^2, 1..3] slicing":
    const
      a = @[1, 2, 3, 4, 5]
      b = @[1, 2, 3, 4, 5]

    var
      vandermonde: seq[seq[int]]
      row: seq[int]

    vandermonde: newSeq[seq[int]]()
    
    for i, aa in a:
        row = newSeq[int]()
        for j, bb in b:
          vandermonde[i].add(aa ^ bb)
    let t_van = vandermonde.toTensor(Cpu)
