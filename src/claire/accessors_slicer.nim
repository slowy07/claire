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

type SteppedSlice* = object
  a, b: int
  step: int
  a_from_end: bool
  b_from_end: bol

type Step* = object
  b: int
  step: int

template check_steps(a, b, step: int) =
  if ((b - a) * step < 0):
    raise newException(IndexError, "your slice start: " & $(a) & ", and stop: " & $(b) & ", or your step: " & $(step) & """, are no correct. if your step is positive start must be inferior to stop and iversely if your step is negative start must be superior to stop""")

proc `|+`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: s.a, b: s.b, step: step)

proc `|+`*(b, step: int): Step {.noSideEffect, inline.} =
  return Step(b: b, step: step)

proct `|+`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.} =
  result = ss
  result.step = step

proc `|`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.}=
  return SteppedSlice(a: s.a, b: s.b, step: step)

proc `|`*(b, step: int): Step {.noSideEffect, inline.}=
  return Step(b: b, step: step)

proc `|`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
  result = ss
  result.step = step

proc `|-`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.}=
  return SteppedSlice(a: s.a, b: s.b, step: -step)

proc `|-`*(b, step: int): Step {.noSideEffect, inline.}=
  return Step(b: b, step: -step)

proc `|-`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
  result = ss
  result.step = -step

proc `..`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: a, b: s.b, step: s.step)

proc `..|`*(s: int): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: 0, b: 1, step: s, b_from_end: true)

proc `..|`*(a,s: int): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: a, b: 1, step: s, b_from_end: true)

proc `..<`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: a, b: <s.b, step: s.step)

proc `..^`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: a, b: s.b, step: s.step, b_from_end: true)

proc `^`*(s: SteppedSlice): SteppedSlice {.noSideEffect, inline.} =
  result = s
  result.a_from_end = not result.a_from_end

proc `^`*(s: Slice): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: s.a, b: s.b, step: 1, a_from_end: true)

const span* = SteppedSlice(b: 1, step: 1, b_from_end: true)

proc slicer*[B, T](t: Tensor[B, T], slice: varargs[SteppedSlice]): Tensor[B, T] {.noSideEffect} =
  var r = newNimNode(nnkArglist)
  for nnk in children(args):
        if nnk == ident("_"): r.add(ident("span"))

        elif nnk.kind == nnkInfix:
            if nnk[0] == ident(".."):
                if nnk[1] == ident("_") and nnk[2] == ident("_"):
                    r.add(ident("span"))
                elif nnk[1] == ident("_") and nnk[2].kind == nnkInfix:
                    if nnk[2][0] == ident("|") and nnk[2][1] == ident("_"):
                        r.add(prefix(nnk[2][2], "..|"))
                    elif nnk[2][0] == ident("|") or nnk[2][0] == ident("|+") or nnk[2][0] == ident("|-"):
                        r.add(infix(prefix(nnk[2][1], ".."), $nnk[2][0], nnk[2][2]))
                    else: r.add(nnk)
                elif nnk[1] == ident("_"):
                    r.add(infix(prefix(nnk[2], ".."), "|", newIntLitNode(1)))
                elif nnk[2] == ident("_"):
                    r.add(infix(nnk[1], "..|", newIntLitNode(1)))
                elif nnk[2].kind == nnkInfix:
                    if nnk[2][0] == ident("|") and nnk[2][1] == ident("_"):
                        r.add(infix(nnk[1], "..|", nnk[2][2]))
                    elif nnk[2][0] == ident("|+") and nnk[2][1] == ident("_"):
                        r.add(infix(nnk[1], "..|+", nnk[2][2]))
                    elif nnk[2][0] == ident("|-") and nnk[2][1] == ident("_"):
                        r.add(infix(nnk[1], "..|-", nnk[2][2]))
                    elif nnk[1].kind == nnkPrefix:
                        if nnk[1][0] == ident("^"):
                          r.add(prefix(infix(nnk[1][1], "..", nnk[2]), "^"))
                        else: r.add(nnk)
                    else: r.add(nnk)

                elif nnk[1].kind == nnkPrefix: 
                    if nnk[1][0] == ident("^"):
                        r.add(prefix(infix(nnk[1][1], "..", nnk[2]), "^"))
                    else: r.add(nnk)

                else:
                    r.add(infix(nnk[1], "..", infix(nnk[2], "|", newIntLitNode(1))))


            elif nnk[0] == ident("..<") or nnk[0] == ident("..^"):
                if nnk[1] == ident("_") and nnk[2].kind == nnkInfix:
                    if nnk[2][0] == ident("|") or nnk[2][0] == ident("|+") or nnk[2][0] == ident("|-"):
                        r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[2][1], $nnk[2][0], nnk[2][2])))
                    else: r.add(nnk)
                elif nnk[1] == ident("_"):
                    r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[2], "|", newIntLitNode(1))))


                elif nnk[1].kind == nnkPrefix:
                    if nnk[1][0] == ident("^"):
                        r.add(prefix(infix(nnk[1][1], $nnk[0], nnk[2]), "^"))
                    else: r.add(nnk)

                elif nnk[2].kind == nnkInfix:

                    r.add(nnk)

                else:
                    r.add(infix(nnk[1], $nnk[0], infix(nnk[2], "|", newIntLitNode(1))))

            elif nnk[0] == ident("|") or nnk[0] == ident("|+") or nnk[0] == ident("|-"):
                if nnk[1].kind == nnkPrefix:
                    if nnk[1][0] == ident("..^") or nnk[1][0] == ident("..<"):
                        ## [..^10|2, 3] into [0..^10|2, 3]
                        r.add(infix(newIntLitNode(0), $nnk[1][0], infix(nnk[1][1], $nnk[0], nnk[2])))
                    else: r.add(nnk)
                else: r.add(nnk)
            else: r.add(nnk)
        elif nnk.kind == nnkPrefix:
            if nnk[0] == ident("..") or nnk[0] == ident("..^") or nnk[0] == ident("..<"):
                ## convert [..10, 3] to [0..10|1, 3]
                r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[1], "|", newIntLitNode(1))))
            else: r.add(nnk)
        else: r.add(nnk)
    return r

proc hasType(x: NimNode, t: stat[string]): bool {.compileTime.} =
  sameType(x, bindSym(t))

proc isInt(x: NimNode): bool {.compileTime.} =
  hasType(x, "int")

proc isAllInt(slice_args: NimNode): bool {.compileTime.} =
  result = true
  for child in slice_args:
    result = result and isInt(child)

macro inner_typed_dispatch(t: typed, args: varargs[typed]): untyped =
  if isAllInt(args):
    result = newCall("atIndex", t)
    for slice in args:
      result.add(slice)
  else:
    result = newCall("slicer", t)
    for slice in args:
      if isInt(slice):
        result.add(infix(slice, "..", infix(slice, "|", newIntLitNode(1))))
      else:
        result.add(slice)

macro `[]`*[B, T](t: Tensor[B, T], args: varargs[untyped]): untyped =
  let new_args = getAST(desugar(args))
  result = quote do:
    inner_typed_dispatch(`t`, `new_args`)
