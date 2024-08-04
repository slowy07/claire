![image_banner](.github/Claire.png)

library used for provided tensor type for deep learning, focus will be on BLAS and GPU support, 1D-tensor, 2D-tensor, 3 and 4D tensor. library will be flexible enough to represent ND-Array, specially on NLP vector

tensor claire currently supporting:
- wrapping any type: string, float, object
- get and set vaue at specific index
- create a tensor from deep nested sequence
- universal function from nim match module like cos, ln, sqrt etc
- create custom universal function with `makeUniversal`, `makeUniversalLocal`, anf `fmap`


`fmap` can even be used on function with input and output of different type. information can be check [here](https://github.com/unicredit/nimblas)

## storage convetion

either C or fortran contiguous are needed for BLAS optimization for tensor of Rank 1 or 2

- `C_contiguous`: `Row Major - default`. last index are fastest change (column in 2D, depth in 3D) - Row (slow), column, depth (fast)
- `F_contiguous`: `Col Mahor`. first index is the fastest change (rows in 2D, depth in 3D)
- `Universal` : any stride

for deep learning on image, depth representing the color channel and change the fastest, row are repersenting another image in a batch and change the slowest. hance C convention is the best and prefered in claire.

## memory consendirations
current CPU cache is 64 byte. tensor data structure at 32 bytes has an ideal size. however, every time retrieve the dimension and strides there is a pointer resolution + bounds checking for a static array. you can see data structure consideration section

## data structure consendirations

dimension and strides have a static size known at runtime. they might the best implemting as VLAs (variable length array) from an indirection point of view. Inconvenient: 2 tensor will not fit in a cache line

ffset is currently a `pointer`. for best safety and reduced complexity it could be an `int`. blas expect a pointer a wrapper function can be used for it. iterating through non-contiguous array (irregulare slice) might be faster with a pointer.

`data` is currently stored in `"seq"` that always deep copy on assignment. taking the transpose of a tensor will allocate new memory (unless optimized away by the compiler)

about quest: should i implementing shallow copy to saving memory (example transpose) or trust in Nim/CUDA GC ?

if are yes
- how to make sure that can modify in-place if shallow copy is allow ?
- how hard will it be to extend claire for others

information: https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html#reference-counting

_if you mis-handle refernece counts you can get problem from memory-leaks to segmentation faults. the only strategy i know of to handle reference counts corretly is blood, sweat and tears_

nim GC perf: https://gist.github.com/dom96/77b32e36b62377b2e7cadf09575b8883

## performance consendirations

adding `OpenMP` pragma for parallel computing on `fmap` and self-implemted BLAS operations.


> [!NOTE]
> EXPERIMENTAL: API may change and going to break
