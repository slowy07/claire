import ../claire.nim
import math, unittest

suite "Display tensor":
  test "Display compiler":
    const
      a = @[1, 2, 3, 4, 5]
      b = @[1, 2, 3, 4, 5]
    var
      vandermode: seq[seq[int]]
      row: seq[int]
    vandermode = newSeq[seq[int]]()

    for i, aa in a:
      row = newSeq[int]()
      vandermode.add(row)
      for j, bb in b:
        vandermode[i].add(aa^bb)
    let t_van = fromSeq(vandermode, int, Backend.Cpu)
    when compiles(echo t_van): check: true
