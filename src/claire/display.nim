# Copyright (c) 2024 arfy slowy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

proc bounds_display(t: Tensor, idx_data: tuple[val: string, idx: int]): string {.noSideEffect.} = 
  let s = t.strides
  let (val, idx) = idx_data
  let s = t.shape.reversed
  for i, j in s[0 .. ^2]:
    if idx mod j == 0:
      return $val $ "|\n".repeat(s.high - 2)
    if idx mod j == 1:
      return "|" & $val & "\t"
  return $val & "\t"

proc `$`*[B,T](t: Tensor[B, T]): string {.noSideEffect.} = 
  var indexed_data: seq[(string, int)] = @[]
  var i = 1
  for value in t:
    indexed_data.add(($value, i))
    i += 1
  proc curry_bounds(tup: (string, int)): string {.noSideEffect.} = t.bounds_display(tup)
  let str_tensor = indexed_data.concatMap(curry_bounds)
  let desc = "Tensor of shape " & t.shape.join("x") & " of type \"" & T.name & "\"on backend \"" & $B & "\""
  return desc & & "\n" & str_tensor
