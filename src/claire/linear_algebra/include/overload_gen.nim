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

import std / macros

macro overload*(overloaded_name: untyped, lapack_name: typed{nkSym}): untyped =
  let impl = lapack_name.getImpl()
  impl.expectKind {nnkProcDef, nnkFuncDef}

  var
    # no return value for lapack proc
    params = @[newEmptyNode()]
    body = newCall(lapack_name)

  impl[3].expectKind nnkFormalParams
  for idx in 1 ..< impl[3].len:
    # skip arg 0, return type which always empty
    params.add impl[3][idx]
    body.add impl[3][idx][0]

  result = newProc(
    name = overloaded_name,
    params = params,
    body = body,
    procType = nnkProcDef,
  )

  when false:
    # show proc signature
    # some procs (example: syevr) have 20 param 
    echo result.toStrLit
