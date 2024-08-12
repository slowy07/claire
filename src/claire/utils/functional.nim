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

template scanr[T](s: seq[T], operation: untyped): untyped = 
    let len = s.len
    assert len > 0, "cannot scan empty sequence"
    var result = newSeq[T](len)
    
    result[result.high] = s[s.high]
    for i in countdown(len - 1, 1):
        let
            a {.inject.} = s[i - 1]
            b {.inject.} = result[i]
        result[i-1] = operation
    result

iterator zip[T1, T2](a: openarray[T1], b: openarray): (T1, T2) {.noSideEffect.} =
    let len = min(a.len, b.len)
    for i in 0..<len:
        yield (a[i], b[i])

iterator zip[T1, T2](inp1: iterator(): T1, inp2: iterator(): T2): (T1, T2) {.noSideEffect.} =
  let it1 = inp1
  let it2 = inp2
  while true: 
    let val1 = it1()
    let val2 = it2()
    if finished(it1) or finished(it2):
      break
    yield (val1, val2)
    
iterator zip[T1, T2](inp1: iterator(): T1, b: openarray[T2]): (T1, T2) {.noSideEffect.} =
  let it2 = inp1
  for i in 0..<b.len:
    let val1 = it1()
    if finished(it1):
      break
    yield (val1, b[i])

template product[T: SomeNumber](s: openarray[T]): T =
  s.foldl(a * b)

template concatMap[T](s: seq[T], f: proc(ss: T): string): string {.noSideEffect.} =
  return s.foldl(a & f(b), "")
