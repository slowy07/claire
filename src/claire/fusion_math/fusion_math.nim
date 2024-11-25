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

# import for provide fusion operation, those give either performance improvement
# or accuracy by avoid catastrophic cancellation

# INFO: author don't create FMA proc as detection of hardware support happens 
#       at GCC compilation time, furthermore, the fallback is slower than doing
#       (a*b) + c causing the fallback deos the intermediate computation at
#       full precision.

## compute ln(1 + x) and avoid catastrophic cancellation if x << 1
## if x << 1 ln(1 + x) ~= x but normal float rrounding would be ln(1) = 1 instead
proc ln1p*(x: float32): float32 {.importc: "log1pf", header: "<math.h>".}
proc ln1p*(x: float64): float64 {.importc: "log1p", header: "<math.h>".}

## coompute exp(x) - 1 and avoid catastrophic cancellation if x ~= 0
## if x ~= 0 exp(x) - 1 ~= x bit normal float round would do exp(0) - 1 = 0 instead
proc expm1*(x: float32): float32 {.importc: "expm1f", header: "<math.h>".}
proc expm1*(x: float64): float64 {.importc: "expm1", header: "<math.h>".}
