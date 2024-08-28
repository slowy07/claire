![image_banner](.github/Claire.png)

library used for provided tensor type for deep learning, focus will be on BLAS and GPU support, 1D-tensor, 2D-tensor, 3 and 4D tensor. library will be flexible enough to represent ND-Array, specially on NLP vector

tensor claire currently supporting:
- wrapping any type: string, float, object
- get and set value at specific index
- create a tensor from deep nested sequence
- universal function from nim match module like cos, ln, sqrt etc
- create custom universal function with `makeUniversal`, `makeUniversalLocal`, anf `fmap`

## Claire Feature
- how to implement integer matrix multiplication and matrix-vector multiplication
    - convert to `float64`, use `BLAS`, convert back to `int`. no issue for `int32` has them all. `int64` may lose precision
    - implement tensor comprehension macro, it may be able to leverage mitems instead of `result[i,j] = alpha * (i - j) * (i + j)`
    - implement cache matrix multiplication.
- how to implementing non-contiguous matrix multiplication and matrix-vector multiplication
    - cache and any stride generic matrix multiplication
    - convert the tensor to C major layout with the items `iterator`


`fmap` can even be used on function with input and output of different type. information can be check [here](https://github.com/unicredit/nimblas)

## storage convention

either C or fortran contiguous are needed for BLAS optimization for tensor of Rank 1 or 2

- `C_contiguous`: `Row Major - default`. last index are fastest change (column in 2D, depth in 3D) - Row (slow), column, depth (fast)
- `F_contiguous`: `Col Mahor`. first index is the fastest change (rows in 2D, depth in 3D)
- `Universal` : any stride

for deep learning on image, depth representing the color channel and change the fastest, row are repersenting another image in a batch and change the slowest. hance C convention is the best and preferred in claire.

## data structure consideration

runtime determines the static size of strides and shape. from an indirection persepective, it might be ideal to build them as variable `length` arrays, or `VLAs`. two tensor cannot fit on cache line, which is inconvenient.


for the time being, `shallowCopy` will only be utilized in certain locations, such as when we wish to modify the original reference while using a different striding scheme in `slicerMut`. there is not view return from slicing. the compiler is capable of the following optimization, in contrast to python:
- coppy elision
- move on assignment
- detect if the original tensor is not used anymore and the copy is unneeded

## memory consendirations
current CPU cache is 64 byte. tensor data structure at 32 bytes has an ideal size. however, every time retrieve the dimension and strides there is a pointer resolution + bounds checking for a static array. you can see data structure consideration section

reference: [Copy semantic](https://forum.nim-lang.org/t/1793/5) "parameter passing doesn't copy, `var x = foo()` doesn't copy but moves `let x = y` doesn't copy but moves, `var x = y` does copy but it can use shallowCopy instead of `=` for that"

in depth ([information](http://blog.stablekernel.com/when-to-use-value-types-and-reference-types-in-swift)) for swift but applicable): performance, safety, and usability


## data structure consendirations

dimension and strides have a static size known at runtime. they might the best implementing as VLAs (variable length array) from an indirection point of view. Inconvenient: 2 tensor will not fit in a cache line

ffset is currently a `pointer`. for best safety and reduced complexity it could be an `int`. blas expect a pointer a wrapper function can be used for it. iterating through non-contiguous array (irregulare slice) might be faster with a pointer.

`data` is currently stored in `"seq"` that always deep copy on assignment. taking the transpose of a tensor will allocate new memory (unless optimized away by the compiler)

about quest: should i implementing shallow copy / ref seq or object for save memory (example transpose) or trust in Nim / CUDA to make proper memory optimization?

if are yes
- how to make sure we can modify in-place if shallow copy is allowed or a ref seq/object is used?
- to avoid reference counting, would it to better to always copy-on-writte, in that case wouldn't it be better to pay the cst upfront on assignment?
- how hard will it be to code claire to avoid cause copy-on-writte was missed

information: https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html#reference-counting

_if you mis-handle reference counts you can get problem from memory-leaks to segmentation faults. the only strategy i know of to handle reference counts correctly is blood, sweat and tears_

nim GC perf: https://gist.github.com/dom96/77b32e36b62377b2e7cadf09575b8883

## coding style

prefer when to use compile-time evaluation. Allow the compiler to do its work. Use `proc` without the inline tag whenever possible. using template if `proc` fails or to access an bject field. adding macro as a last option to change the AST tree or rewrite code. readiblity, maintainability, and performance are extremely important (in no particular order). when you don't require side effect or an iterator, use functional techniques like map and scanr instead of or loops.

## performance consendirations

adding `OpenMP` pragma for parallel computing on `fmap` and self-implemted BLAS operations.

## CUDA consideration

all init procedures now take a backend argument (`CPU`, `CUDA`, etc). if backend contains function called `zeros`, `ones`, or `randomTensor`, it will eliminate the need to generate a tensor on the CPU and then transfer it to the backend. the drawback is that the usage auto return types, untyped templates, and when t is tensor complicates the procedures. having a rewrite rule to convert `randomTensor(...).toCuda()` into the straight cuda function is an alternative.


> [!NOTE]
> EXPERIMENTAL: API may change and going to break
