import ../claire
import unittest

suite "accessing, testing tensor value":
    test "access, testing single value":
        var a = newTensor(@[2, 3, 4], int, Backend.Cpu)
        a[1, 2, 2] = 122
        check: a[1, 2, 2] == 122
        
    test "out of bound check":
        var a = newTensor(@[2, 3, 4], int, Backend.Cpu)
        when compiles(a[2, 0, 0] = 200): false