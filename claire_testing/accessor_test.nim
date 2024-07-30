import ../claire
import unittest

suite "accessing, testing tensor value":
    test "access, testing single value":
        var a = newTensor(@[2, 3, 4], int, Backend.Cpu)
        a[1, 2, 2] = 122
        check: a[1, 2, 2] == 122
        # echo a
        
    var b = newTensor(@[3, 4], int, Backend.Cpu)
    b[1, 2] = 12
    check: b[1, 2] == 12
    b[0, 0] = 999
    check: b[0, 0] == 999
    b[2, 3] = 111
    check: b[2, 3] == 111
    b[2, 0] = 555
