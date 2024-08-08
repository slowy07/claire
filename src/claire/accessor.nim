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

proc check_index(t: Tensor, idx: varargs[int]) {.noSideEffect.} =
  if idx.len != t.rank:
    raise newException(IndexError, "number of arguments: " & $(idex.len) & ", is different from tensor rank: " & $(t.rank))


template getIndex[B: static[Backend], T](t: Tensor[B, T], idx: varargs[int]): int {.used.} = 
    when compileOption("boundChecks"):
      t.check_index(idx)
    var real_idx = t.offset
    for i, j in zip(t.strides, idx):
        real_idx += i * j
      return real_idx

proc atIndex*[B: static[Backend], T](t: Tensor[B, T], idx: varargs[int]): T {.noSideEffect.} =
  return t.data[t.getIndex(idx)]

proc `atIndexMut`*[B: static[Backend], T](t: var Tensor[B, T], idx: varargs[int], val: T) {.noSideEffect.} =
  t.data[t.getIndex(idx)] = val

type
  IterKind = enum Values, MemOffset, ValCoord, ValMemOffset # Coord

template strided_iteration[B, T](t: Tensor[B, T], strider: IterKind): untyped =
  var coord = newSeq[int](t.rank)
  var backstrides: seq[int] = @[]
  for i, j in zip(t.strides, t.shape): backstrides.add(i*(j-1))
  
  var iter_pos = t.offset

  for i in 0..<t.shape.product:
    when strider == IterKind.Values: yield t.data[iter_pos]
    elif strider == IterKind.ValCoord: yield (t.data[iter_pos], coord)
    elif strider == IterKind.MemOffset: yield iter_pos
    elif strider == IterKind.ValMemOffset: yield (t.data[iter_pos], iter_pos)

    for k in countdown(t.rank - 1, 0):
      if coord[k] < t.shape[k]-1:
          coord[k] += 1
          iter_pos += t.strides[k]
          break
      else:
        coord[k] = 0
        iter_pos -= backstrides[k]

iterator items*[B, T](t: Tensor[B,T]): T {.noSideEffect.} =
  t.strided_iteration(IterKind.Values)

proc values*[B, T](t: Tensor{[B, T]): auto {.noSideEffect.} =
  return iterator(): T = t.strided_iteration(IterKind.Values)

iterator pairs*[B, T](t: Tensor[B, T]): (T, seq[int]) {.noSideEffect.} =
  t.strided_iteration(IterKind.ValCoord)

iterator real_indices(t: Tensr): int {.noSideEffect.} =
  t.strided_iteration(IterKind.MemOffset)

proc real_indices(t: Tensor): auto {.noSideEffect.} =
  return iterator(): int = t.strided_iteration(IterKind.MemOffset)
