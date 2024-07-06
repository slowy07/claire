import ../claire.nim
import unittest, math, sequtils

suite "creating new tensor":
  test "create from sequence":
    let t1 = fromSeq(@[1, 2, 3], int, Backend.Cpu)
    check: t1.shape == @[3]
    check: t1.rank == 1

    const
      a = @[1, 2, 3, 4, 5]
      b = @[1, 2, 3, 4, 5]

    var
      vandermonde: seq[seq[int]]
      row: seq[int]

    vandermonde = newSeq[seq[int]]()
    for i, aa in a:
        row = newSeq[int]()
        vandermonde.add(row)
        for j, bb in b:
          vandermonde[i].add(aa^bb)

    let t2 = fromSeq(vandermonde, int, Backend.Cpu)
    check: t2.rank == 2
    check: t2.shape == @[5, 5]

    let nest3 = @[
      @[
        @[1, 2, 3],
        @[1, 2, 3]
      ],
      @[
        @[3, 2, 1]
        @[3, 2, 1]
      ],
      @[
        @[4, 4, 5],
        @[4, 4, 4]
      ]
    ]
    let t3 = fromSeq(nest3, int, Backend.Cpu)
    check: t3.rank == 3
    check: t3.shape == @[3, 2, 3]

  test "check tensor shape is row by-column order":
    let s = @[@[1, 2, 3], @[3, 2, 1]]
    let t = from fromSeq(s, int, Backend.Cpu)
    check: t.shape == @[3, 2]
