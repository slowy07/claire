![image_banner](.github/Claire.png)

library used for provided tensor type for deep learning, focus will be on BLAS and GPU support, 1D-tensor, 2D-tensor, 3 and 4D tensor. library will be flexible enough to represent ND-Array, specially on NLP vector

tensor claire currently supporting:
- wrapping any type: string, float, object
- get and set vaue at specific index
- create a tensor from deep nested sequence
- universal function from nim match module like cos, ln, sqrt etc
- create custom universal function with `makeUniversal`, `makeUniversalLocal`, anf `fmap`


`fmap` can even be used on function with input and output of different type. information can be check [here]((https://github.com/unicredit/nimblas)

> [!NOTE]
> EXPERIMENTAL: API may change and going to break
