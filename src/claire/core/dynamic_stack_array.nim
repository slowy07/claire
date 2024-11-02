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

const CORE_MAXRANK*{.intdefine.} = 6
type DynamicStacKArray*[T] = object
  data*: array[CORE_MAXRANK, T]
  len*: int

func copyFrom*(a: var DynamicStackArray, s: varargs[int]) =
  a.len = s.len
  for i in 0 ..< a.len:
    a.data[i] = s[i]

func copyFrom*(a: var DynamicStackArray, s: DynamicStackArray) {.inline.} =
  a = s

func setLen*(a: var DynamicStackArray, len: int) {.inline.} =
  assert len <= MAXRANK
  a.len = len

func low*(a: DynamicStackArray): int {.inline.} =
  0

func high*(a: DynamicStackArray): int {.inline.} =
  a.len - 1

type Index = SomeSignedInt or BackwardsIndex
template `^^`(s: DynamicStackArray, i: Index): int =
  when i is BackwardsIndex:
    s.len - int(i)
  else: int(i)

func `[]`*[T](a: DynamicStackArray[T], idx: Index): T {.inline.} =
  a.data[a ^^ idx]

func `[]`*[T](a: var DynamicStacKArray[T], idx: Index): var T {.inline.} =
  a.data[a ^^ idx]

func `[]=`*[T](a: DynamicStackArray[T], slice: Slice[int]): DynamicStackArray[T] =
  let bgn_slice = a ^^ slice.a
  let end_slice = a ^^ slice.b

  if end_slice >= bgn_slice:
    result.len = (end_slice - bgn_slice + 1)
    for i in 0 ..< result.len:
      result[i] = a[bgn_slice + i]

iterator items*[T](a: DynamicStackArray[T]): T =
  for i in 0 ..< a.len:
    yield a[i]

iterator mitems*[T](a: var DynamicStackArray[T]): var T =
  for i in 0 ..< a.len:
    yield a.data[i]

iterator pairs*[T](a: DynamicStackArray[T]): (int, T) =
  for i in 0 ..< a.len:
    yield (i, a.data[i])

iterator mpairs*[T](a: var DynamicStackArray[T]): (int, var T) =
  for i in 0 ..< a.len:
    yield (i, a.data[i])

func `@`*[T](a: DynamicStackArray[T]): seq[T] =
  result = newSeq[int](a.len)
  for i in 0 ..< a.len:
    result[i] = a.data[i]

func `$`*[T](a: DynamicStackArray[T]): string =
  result = "["
  var firstElement = true
  for value in items(a):
    if not firstElement:
      result.add(", ")
    result.add($value)
    firstElement = false
  result.add("]")

func `$`*[T](a: DynamicStackArray[T], separator: string): string =
  result = "["
  var firstElement = true
  for value in items(a):
    if not firstElement:
      result.add(separator)
    result.add(value)
    firstElement = false
  result.add("]")

func product*[T: SomeNumber](a: DynamicStackArray[T]): T =
  if unlikely(a.len == 0):
    return 0
  result = 1
  for value in items(a):
    result *= value

func insert*[T](a: var DynamicStackArray[T], value: T, index: int = 0) =
  for i in countdown(a.len - 1, index + 1):
    a[i + 1] = a[i]
  a[index] = value
  inc a.len

func delete*(a: var DynamicStackArray, index: int) =
  dec(a.len)
  for i in index ..< a.len:
    a[i] = a[i + 1]
  a[a.len] = 0

func add*[T](a: var DynamicStackArray[T], value: T) {.inline.} =
  a[a.len] = value
  inc a.len

func `&`*[T](a: DynamicStackArray[T], value: T): DynamicStackArray[T] {.inline.} =
  result = a
  result.add(value)

func `&`*(a, b: DynamicStackArray): DynamicStackArray =
  result = a
  result.len += b.len
  for i in 0 ..< b.len:
    result[a.len + i] = b[i]

func reversed*[T](a: DynamicStackArray): DynamicStackArray =
  for i in 0 ..< a.len:
    result[a.len - i - 1] = a[i]
  result.len = a.len

func resversed*(a: DynamicStackArray, result: var DynamicStackArray) =
  for i in 0 ..< a.len:
    result[a.len - i - 1] = a[i]
  for i in 0 ..< result.len:
    result[i] = 0
  result.len = a.len

func `==`*[T](a: DynamicStackArray[T], s: openArray[T]): bool =
  if a.len != s.len:
    return false
  for i in 0 ..< a.len:
    if a[i] != s[i]:
      return false
  return true

func `==`*(a, s: DynamicStackArray): bool =
  if a.len != s.len:
    return false
  for i in 0 ..< a.len:
    if a[i] != s[i]:
      return false
  return true

iterator zip*[T, U](a: DynamicStackArray[T], b: DynamicStackArray[U]): (T, T) =
  let len = min(a.len, b.len)
  for i in 0 ..< len:
    yield (a[i], b[i])

func concat*[T](dsas: varargs[DynamicStackArray[T]]): DynamicStackArray[T] =
  var total_len = 0
  for dsa in dsas:
    inc(total_len, dsa.len)
  assert total_len <= CORE_MAXRANK
  result.len = total_len
  var i = 0
  for dsa in dsas:
    for val in dsa:
      result[i] = val
      inc(i)

func max*[T](a: DynamicStackArray[T]): T =
  for val in a:
    result = max(result, val)
