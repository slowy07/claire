import os, strutils
import ../claire

proc divmod[T: SomeInteger](n: T, b: T): (T, T) =
  return (n div b, n mod b)

proc matgen(n: int): auto =
  result = newTensr(@[n, n], float64, Backend.Cpu)
  let tmp = 1.0 / (n * n).float64
  var counter = 0
  for val in result.mitems:
    let (i, j) = counter.divmod(n)
    val = (i - j).float64 * (i + j).float64 * tmp
    inc counter

var n = 100
if paramCount() > 0:
  n = parseInt(paramStr(1))
n = n div 2 * 2

let a, b = matgen n
let c = a * b
echo formatFloat(c[n div 2, n div 2], ffDefault, 8)
