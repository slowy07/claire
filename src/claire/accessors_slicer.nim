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

proc check_steps(a, b, step: int) {.noSideEffect.} =
  if ((b - a) * step < 0):
    raise newException(IndexError, "your slice start: " & $a & ", and stop: " & $b & ", or your step: " & $step & """, are no correct. if your step is positive start must be inferior to stop and iversely if your step is negative start must be superior to stop""")

proc check_shape(a, b: Tensor | openarray) {.noSideEffect.} =
  if a.shape != b.shape:
    raise newException(IndexError, "your tensor or openarray do not have the same shape: " & $a.shape.join("x") & " and " & $b.shape.join("x"))

proc `|`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: s.a, b: s.b, step: step)

proc `|`*(b, step: int): Step {.noSideEffect, inline.}=
  return Step(b: b, step: step)

proc `|`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
  result = ss
  result.step = step

proc `|+`*(s: Slice[int], step: int): SteppedSlice{.noSideEffect, inline.} =
  return `|`(s, step)

proc `|+`*(b, step: int): Step {.noSideEffect, inline.} =
  return `|`(b, step)

proc `|+`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.} =
  return `|`(ss, step)

proc `|-`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.}=
  return SteppedSlice(a: s.a, b: s.b, step: -step)

proc `|-`*(b, step: int): Step {.noSideEffect, inline.}=
  return Step(b: b, step: -step)

proc `|-`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
  result = ss
  result.step = -step

proc `..`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
  return SteppedSlice(a: a, b: s.b, step: s.step)

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

macro desugar(args: untyped): typed =
  var r = newNimNode(nnkArglist)

  for nnk in childern(args):
    let nnk_joker = nnk == ident("_")

    let nnk0_inf_dotdot = (
      nnk.kind == nnkInfix and nnk[0] == ident("..")
    )
    let nnk0_inf_dotdot_alt = (
      nnk.kind == nnkInfix and (
        nnk[0] == ident("..<") or nnk[0] == ident("..^")
      )
    )

    let nnk_inf_dotdot_all = (
      nnk0_inf_dotdot or nnk0_inf_dotdot_alt
    )

    let nnk0_pre_hat = (
      nnk.kind == nnkPrefix and nnk[0] == ident("^")
    )

    let nnk1_joker = (
      nnk.kind == nnkInfix and nnk[1] == ident("_")
    )

    let nnk10_hat = (
      nnk.kind == nnkInfix and nnk[1].kind == nnkPrefix and nnk[1][0] == ident("^")
    )

    let nnk2_joker = (
      nnk.kind == nnkInfix and nnk[2] == ident("_")
    )

    let nnk20_bar_pos = (
      nnk.kind == nnkInfix and nnk[2].kind == nnkInfix and (
        nnk[2][0] == ident("|") or nnk[2][0] == ident("|+")
      )
    )

    let nnk20_bar_min = (
      nnk.kind == nnkInfix and nnk[2].kind == nnkInfix and nnk[2][0] == ident("|-")
    )

    let nnk20_bar_all = nnk20_bar_pos or nnk20_bar_min
    
    let nnk21_joker = (
      nnk.kind == nnkInfix and nnk[2].kind == nnkInfix and nnk[2][1] == ident("_")
    )

    if nnk_joker:
      r.add(ident("span"))
    elif nnk0_inf_dotdot and nnk1_joker and nnk2_joker:
      r.add(ident("span"))
    elif nnk0_inf_dotdot and nnk1_joker and nnk20_bar_all and nnk21_joker:
      r.add(infix(newIntLitNode(0), "..^", infix(newIntLitNode(1), $nnk[2][0], nnk[2][2])))
    elif nnk0_inf_dotdot_all and nnk1_joker and nnk20_bar_all:
      r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[2][1], $nnk[2][0], nnk[2][2])))
    elif nnk0_inf_dotdot_all and nnk1_joker:
      r.add(infix(newIntLitNode(0), $nnk[0], infix(nnk[2], "|", newIntLitNode(1))))
    elif nnk0_inf_dotdot and nnk2_joker:
      r.add(infix(nnk[1], "..^", infix(newIntLitNode(1), "|", newIntLitNode(1))))
    elif nnk0_inf_dotdot and nnk20_bar_pos and nnk21_joker:
      r.add(infix(nnk[1], "..|", nnk[2][2]))
    elif nnk0_inf_dotdot and nnk20_bar_min and nnk21_joker:
      raise newException(IndexError, "please use explicit end of range instead of `_` when the stape are negative" )
    elif nnk0_inf_dotdot_all and nnk10_hat and nnk20_bar_all:
      r.add(prefix(infix(nnk[1][1], $nnk[0], nnk[2]), "^"))
    elif nnk0_inf_dotdot_all and nnk10_hat:
      r.add(prefix(infix(nnk[1][1], $nnk[0], infix(nnk[2],"|",newIntLitNode(1))), "^"))
    elif nnk0_inf_dotdot_all and nnk20_bar_all:
      r.add(nnk)
    elif nnk0_inf_dotdot_all:
      r.add(infix(nnk[1], $nnk[0], infix(nnk[2], "|", newIntLitNode(1))))
    elif nnk0_pre_hat:
      r.add(prefix(infix(nnk[1], "..^", infix(nnk[1],"|",newIntLitNode(1))), "^"))
    else:
      r.add(nnk)
    
  return r

template slicerT[B, T](result: Tensor[B, T], slices: varargs[SteppedSlice]): untyped =
  for i, slice in slices:
      let a = if slice.a_from_end: result.shape[i] - slice.a
              else: slice.a
      let b = if slice.b_from_end: result.shape[i] - slice.b
              else: slice.b

      when compileOption("boundChecks"): check_steps(a, b, slice.step)
      result.offset += a * result.strides[i]
      result.strides[i] *= slice.step
      result.shape[i] = abs((b - a) div slice.step) + 1

proc slicer*[B, T](t: Tensor[B, T], slices: varargs[SteppedSlice]): Tensor[B, T] {.noSideEffect.} =
  result = t
  for i, slice in slices:
    let a = if slice.a_from_end: result.shape[i] - slice.a
            else: slice.a
    let b = if slice.b_from_end: result.shape[i] - slice.b
            else: slice.b

    when compileOption("boundChecks"): check_steps(a, b, slice.step)
    result.offset += a * result.strides[i]
    result.strides[i] *= slice.step
    result.shape[i] = abs((b-a) div slice.step) + 1

proc shallowSlicer[B, T](t: Tensor[B, T], slices: varargs[SteppedSlice]): Tensor[B, T] {.noSideEffect.} =
  result = shallowCopy(t)
  slicerT(result, slices)

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

proc slicerMut*[B, T](t: var Tensor[B, T], slices: varargs[SteppedSlice], val: T) {.noSideEffect.} =
  let sliced = t.slicer(slices)
  for real_idx in sliced.real_indices:
    t.data[real_idx] = val

proc slicerMut*[B, T](t: var Tensor[B, T], slices: varargs[SteppedSlice], oa: openarray[T]) =
  let sliced = t.slicer(slices)
  when compileOption("boundChecks"):
    check_shape(sliced, oa)

  let data = toSeq(flatIter(oa))
  when compileOption("boundChecks"):
    check_nested_elements(oa.shape, data.len)

proc slicerMut*[B1, B2, T](t: var Tensor[B1, T], slices: varargs[SteppedSlice], t2: Tensor[B2, T]) {.noSideEffect.} =
  let sliced = t.slicer(slices)
  when compileOption("boundChecks"): check_shape(sliced, t2)
  for real_idx, val in zip(sliced.real_indices, t2.values):
    t.data[real_idx] = val

macro inner_typed_dispatch_mut(t: typed, args: varargs[typed], val: typed): untyped =
  if isAllInt(args):
    result = newCall("atIndexMut", t)
    for slice in args:
      result.add(slice)
    result.add(val)
  else:
    result = newCall("slicerMut", t)
    for slice in args:
      if isInt(slice):
        result.add(infix(slice, "..", infix(slice, "|", newIntLitNode(1))))
      else:
        result.add(slice)

macro `[]=`*[B, T](t: var Tensor[B, T], args: varargs[untyped]): untyped =
  var tmp = args
  let var = tmp.pop
  let new_args = getAST(desugar(tmp))

  result = quote do:
    inner_typed_dispatch_mut(`t`, `new_args`, `val`)
