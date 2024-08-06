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
  result = t
  for i, slice in slices:
    let a = if slice.a_from_end: result.shape[i] - slice.a else: slice.a
    let b = if slice.b_from_end: result.shape[i] - slice.b else: slice.b
    
    when compileOption("boundChecks"): check_steps(a, b, slice.step)
    result.offset += a * result.strides[i]
    result.strides[i] *= slice.step
    result.shape[i] = abs((b - a) div slice.step) + 1

macro desugar(args: untyped): typed =
  var r = newNimNode(nnkArglist)

  for nnk in childern(args):
    let nnk_joker = nnk == ident("_")
    let nnk0_inf_dotdot = if nnk.kind == nnkInfix: nnk[0] == ident("..") else: false
    let nnk0_inf_dotdot_alt = if nnk.kind == nnkInfix: nnk[0] == ident("..<") or nnk[0] == ident("..^") else: false
    let nnk0_inf_dotdot_all = nnk0_inf_dotdot or nnk0_inf_dotdot_alt
    let nnk0_inf_bar_all = if nnk.kind == nnkInfix: nnk[0] == ident("|") or nnk[0] == ident("|+") or nnk[0] == ident("|-") else: false
    let nnk0_pre_dotdot_all =  if nnk.kind == nnkPrefix: nnk[0] == ident("..") or nnk[0] == ident("..<") or nnk[0] == ident("..^") else: false
    let nnk0_pre_hat_all = if nnk.kind == nnkPrefix: nnk[0] == ident("^")
                           else: false
    let nnk1_joker = if nnk.kind == nnkInfix: nnk[1] == ident("_") else: false
    let nnk10_hat = if nnk.kind == nnkInfix:
                      if nnk[1].kind == nnkPrefix: nnk[1][0] == ident("^")
                      else: false
                    else: false
    let nnk10_dotdot_pre_alt = if nnk.kind == nnkInfix:
                                  if nnk[1].kind == nnkPrefix:
                                    nnk[1][0] == ident("..^") or nnk[1][0] == ident("..<")
                                  else: false
                              else: false
    let nnk2_joker = if nnk.kind == nnkInfix: nnk[2] == ident("_") else: false
    let nnk20_bar_pos = if nnk.kind == nnkInfix:
                          if nnk[2].kind == nnkInfix: nnk[2][0] == ident("|") or nnk[2][0] == ident("|+")
                          else: false
                        else: false

    let nnk20_bar_min = if nnk.kind == nnkInfix:
                          if nnk[2].kind == nnkInfix: nnk[2][0] == ident("|-")
                          else: false
                        else: false

    let nnk20_bar_all = nnk20_bar_pos or nnk20_bar_min
    let nnk21_joker = if nnk.kind == nnkInfix:
                        if nnk[2].kind == nnkInfix: nnk[2][1] == ident("_")
                        else: false
                      else: false

    if nnk_joker:
      r.add(ident("span"))
    elif nnk0_inf_dotdot and nnk1_joker and nnk2_joker:
      r.add(ident("span"))
    elif nnk0_inf_dotdot and nnk1_joker and nnk20_bar_min and nnk21_joker:
      r.add(prefix(nnk[2][2], "..|-"))
    elif nnk0_inf_dotdot_all and nnk1_joker and nnk20_bar_all:
      r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[2][1], $nnk[2][0], nnk[2][2])))
    elif nnk0_inf_dotdot_all and nnk1_joker:
      r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[2], "|", newIntLitNode(1))))
    elif nnk0_inf_dotdot and nnk2_joker:
      r.add(infix(nnk[1], "..|", newIntLitNode(1)))
    elif nnk0_inf_dotdot and nnk20_bar_pos and nnk21_joker:
      r.add(infix(nnk[1], "..|", nnk[2][2]))
    elif nnk0_inf_dotdot and nnk20_bar_min and nnk21_joker:
      raise newException(IndexError, "please use explicit end of range " & "instead of `_` when the steps are negative")
    elif nnk0_inf_dotdot_all and nnk10_hat and nnk20_bar_all:
      r.add(prefix(infix(nnk[1][1], $nnk[0], nnk[2]), "^"))
    elif nnk0_inf_dotdot_all and nnk20_bar_all:
      r.add(nnk)
    elif nnk0_inf_dotdot_all:
      r.add(infix(nnk[1], $nnk[0], infix(nnk[2], "|", newIntLitNode(1))))
    elif nnk0_inf_bar_all and nnk10_dotdot_pre_alt:
      r.add(infix(newIntLitNode(0), $[1][0], infix(nnk[1][1], $nnk[0], nnk[2])))
    elif nnk0_pre_dotdot_all:
      r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[1], "|", newIntLitNode(1))))
    elif nnk0_pre_hat_all:
      r.add(prefix(infix(nnk[1], "..^", infix(nnk[1], "|", newIntLitNode(1))), "^"))
    else:
      r.add(nnk)
    
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
