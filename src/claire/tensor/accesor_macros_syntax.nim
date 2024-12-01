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
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY.

import ../claire_third/dynamic_stack_array


## internal: slice object related tensor single dim
##           - a, b: respectively the beginning and the end of the range of the dimension
##           - step: the stepping of the slice
##           - a/b: indicate if a/b should be counted from 0 or from end of the tensor
##                  relevan dimension
type SteppedSlice* = object
  a*, b*: int
  step*: int
  a_from_end*: bool
  b_from_end*: bool

## internal: workaround to build ``steppedslice`` without using parenthesis
type Step* = object
  b: int
  step: int

const _* = SteppedSlice(b: 1, step: 1, b_from_end: true)
type Ellipsis* = object
const `...`* = Ellipsis()
type ArrayOfSlices* = DynamicStackArray[SteppedSlice]

proc toArrayOfSlices*(s: varargs[SteppedSlice]): ArrayOfSlices {.inline.} =
  result.len = s.len
  for i in 0..<s.len:
    result.data[i] = s[i]

proc initSpanSlices*(len: int): ArrayOfSlices {.inline.} =
  result.len = len
  for i in 0..<len:
    result.data[i] = SteppedSlice(b: 1, step: 1, b_from_end: true)

proc `|`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: s.a, b: s.b, step: step)

proc `|`*(b, step: int): Step {.noSideEffect, inline.} =
  return Step(b: b, step: step)

proc `|`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.} =
  result = ss
  result.step = step

proc`|+`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.} =
  return `|`(s, step)

proc `|+`*(b, step: int): Step {.noSideEffect, inline.} =
  return `|`(b, step)

proc `|+`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.} =
  return `|`(ss, step)

proc `|-`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: s.a, b: s.b, step: -step)

proc `|-`*(b, step: int): Step {.noSideEffect, inline.} =
  return Step(b: b, step: -step)

proc `|-`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.} =
  result = ss
  result.step = -step

proc `..`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: a, b: s.b, step: s.step)

proc `..<`*(a: int, s: step): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: b, b: s.b - 1, step: s.step)

proc `..^`(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: a, b: s.b, step: s.step, b_from_end: true)

proc `^`*(s: SteppedSlice): SteppedSlice {.noSideEffect, inline.} =
  result = s
  result.a_from_end = not result.a_from_end

proc `^`*(s: Slice): SteppedSlice {.noSideEffect, inline.} =
  return steppedslice(a: s.a, b: s.b, step: 1, a_from_end: true)
