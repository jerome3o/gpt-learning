# MPT-7B

Attempt at playing around with MPT-7B

* [Release](https://www.mosaicml.com/blog/mpt-7b)
* [Model](https://huggingface.co/mosaicml/mpt-7b-instruct)


# Issues encountered

* AMD GPUs: Tensors loaded onto meta device? see [fake tensors](https://pytorch.org/torchdistx/latest/fake_tensor.html)
    * not sure why. Trying running it without alibi.
