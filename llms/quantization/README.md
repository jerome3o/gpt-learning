# Getting LLM.int8() Quantisation working on AMD GPUs

* Need to get [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) package working with ROCm
* bronctoc's rocm fork [here](https://github.com/broncotc/bitsandbytes-rocm)
    * No docs on how to get it to build
* jinsihou19's fork has a make command for hip [here](https://github.com/jinsihou19/bitsandbytes-rocm)
    * needs [hipblas](https://github.com/ROCmSoftwarePlatform/hipBLAS)
    * needs cmake and gfortran (`sudo apt install cmake gfortran`)
    * hipblas needs [rocblas](https://github.com/ROCmSoftwarePlatform/rocBLAS)
