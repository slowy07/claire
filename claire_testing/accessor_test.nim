import ../claire
import unittest, math

suite "accessing, testing tensor value":
    test "access, testing single value":
        var a = newTensor(@[2, 3, 4], int, Backend.Cpu)
        a[1, 2, 2] = 122
        check: a[1, 2, 2] == 122
        
        var b = newTensor(@[3, 4], int, Backend.Cpu)
        b[1, 2] = 12
        check: b[1, 2] == 12
        b[0, 0] = 999
        check: b[0, 0] == 999
        b[2, 3] = 111
        check b[2, 3] == 111

    test "out of bound check":
      var a = newTensor(@[2, 3, 4], int, Backend.Cpu)
      expect(IndexError):
        a[2, 0, 0] = 200

      var b = newTensor(@[3, 4], int, Backend.Cpu)
      expect(IndexError):
        b[3, 4] = 999
      expect(IndexError):
        discard b[-1, 0]
      expect(IndexError):
        discard b[0, -2]

    test "iterator":
      const
        a = @[1, 2, 3, 4, 5]
        b = @[1, 2, 3]

      var
        vd: seq[seq[int]]
        row: seq[int]

      vd = newSeq[seq[int]]()
      for i, aa in a:
        row = newSeq[int]()
        vd.add(row)
        for j, bb in b:
          vd[i].add(aa^bb)

      let nda_vd = fromSeq(vd, int, Backend.Cpu)
      let expected_seq = @[1,1,1,2,4,8,3,9,27,4,16,64,5,25,125]
      var seq_val: seq[int] = @[]
      for i in nda_vd:
          seq_val.add(i)

      check: seq_val == expected_seq
      
      var seq_validx: seq[tuple[val: int, idx: seq[int]]] = @[]
      for i, j in nda_vd:
          seq_validx.add((i, j))
        
      check: seq_validx[0] == (1, @[0, 0])
      check: seq_validx[10] == (16, @[3, 1])

      let t_nda = transpose(nda_vd)

      var seq_transpose: seq[tuple[val: int, idx: seq[int]]] = @[]
      for i, j in t_nda:
        seq_transpose.add((i, j))

      check: seq_transpose[0] == (1, @[0, 0])
      check: seq_transpose[8] == (16, @[1, 3])
