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
    vandermonde.add(row)
    for j, bb in b:
      vandermonde[i].add(aa^bb)

  let t_van = vandermonde.toTensor(Cpu)

  test "basic indexing":
    check: t_van[2, 3] == 81

  test "basic indexing [1+1, 2*2*1]":
    check: t_van[1+1, 2*2*1] == 243

  test "slice from the end - expect non-negative step error":
    expect(indexError):
      discard t_van[^1..0, 3]

suite "slice mutation":
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

  let t_van_immut = vandermonde.toTensor(Cpu)

  test "immutable - let variable cannot be changed":
    when compiles(t_van_immut[1..2, 3..4] = 99):
      check false
    when compiles(t_van_immut[0..1, 0..1] = [111, 222, 333, 444]):
      check false
    when compiles(t_van_immut[0..1, 0..1] = t_van_immut[111, 222, 333, 444]):
      check false

  test "setting a slice to a single value":
    var t_van = t_van_immut
    let test =  @[@[1,  1,   1,   1,    1],
                @[2,  4,   8, 999,  999],
                @[3,  9,  27, 999,  999],
                @[4, 16,  64, 256, 1024],
                @[5, 25, 125, 625, 3125]]
    let t_test = test.toTensor(Cpu)
    t_van[1..2, 3..4] = 999
    check: t_van == t_test

  test "setting a slice to an array/seq of values":
    var t_van = t_van_immut
    let test =  @[@[111,  222,   1,   1,    1],
                @[333,  444,   8,  16,   32],
                @[  3,    9,  27,  81,  243],
                @[  4,   16,  64, 256, 1024],
                @[  5,   25, 125, 625, 3125]]

    let t_test = test.toTensor(Cpu)
    t_van[0..1,0..1] = [111, 222, 333, 444]
    check: t_van == t_test

  test "bounds checking":
    var t_van = t_van_immut
    expect(indexError):
      t_van[0..1, 0..1] = [111, 222, 333, 444, 555]
    expect(indexError):
      t_van[0..1, 0..1] = [111, 222, 333]
    expect(indexError):
      t_van[^2..^1, 2..4] = t_van[1, 4..2|-1]
