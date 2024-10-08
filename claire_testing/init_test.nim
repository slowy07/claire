import ../claire.nim
import unittest, math, sequtils

suite "creating new tensor":
  test "create from sequence":
    let t1 = @[1, 2, 3].toTensor(Cpu)
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

    let t2 = vandermonde.toTensr(Cpu)
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
      ],
      @[
        @[6, 6, 6]
        @[6, 6, 6]
      ]
    ]
    let t3 = nest3.toTensor(Cpu)
    check: t3.rank == 3
    check: t3.shape == @[4, 2, 3]

    let u = @[@[1.0, -1, 2], @[0.0, -1]]
    expect(IndexError):
      discard u.toTensor(Cpu)

  test "check tensor shape is row by-column order":
    let s = @[@[1, 2, 3], @[3, 2, 1]]
    let t = s.toTensor(Cpu)
    check: t.shape == @[2, 3]

    let u = newTensor(@[2, 3], int, Cpu)
    check: u.shape == @[2, 3]
    check: u.shape == t.shape
