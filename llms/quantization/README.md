# Getting LLM.int8() Quantisation working on AMD GPUs

* Need to get [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) package working with ROCm
* bronctoc's rocm fork [here](https://github.com/broncotc/bitsandbytes-rocm)
    * No docs on how to get it to build
* jinsihou19's fork has a make command for hip [here](https://github.com/jinsihou19/bitsandbytes-rocm)
    * needs [hipblas](https://github.com/ROCmSoftwarePlatform/hipBLAS)
    * needs cmake and gfortran (`sudo apt install cmake gfortran`)
    * hipblas needs [rocblas](https://github.com/ROCmSoftwarePlatform/rocBLAS)
        * `sudo apt install rocblas` failed, and broke `apt` (fixed with `sudo apt remove rocblas rocblas-dev`)
        * cloned rocblas, trying to use cmake - it needs version `3.16.8` (I only have `3.16.3`)
            * followed [this](https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu) guide to update cmake
        * rocblas needs `msgpack`, this caused some issues, couldn't figure out how to fix this, but there is an option to install rocblas without `msgpack`, with `./install --no-msgpack`.
