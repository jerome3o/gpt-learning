# Getting LLM.int8() Quantisation working on AMD GPUs

* Need to get [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) package working with ROCm
* bronctoc's rocm fork [here](https://github.com/broncotc/bitsandbytes-rocm)
    * No docs on how to get it to build
* jinsihou19's fork has a make command for hip [here](https://github.com/jinsihou19/bitsandbytes-rocm)
    * needs [hipblas](https://github.com/ROCmSoftwarePlatform/hipBLAS)
        * needs cmake and gfortran (`sudo apt install cmake gfortran`)
        * needs [rocblas](https://github.com/ROCmSoftwarePlatform/rocBLAS)
            * `sudo apt install rocblas` failed, and broke `apt` (fixed with `sudo apt remove rocblas rocblas-dev`)
            * cloned rocblas, trying to use cmake - it needs version `3.16.8` (I only have `3.16.3`)
                * followed [this](https://askubuntu.com/questions/829310/how-to-upgrade-cmake-in-ubuntu) guide to update cmake
                * rocblas needs `msgpack`, this caused some issues, couldn't figure out how to fix this, but there is an option to install rocblas without `msgpack`, with `./install --no-msgpack`.
        * needs [rocsolver](https://github.com/ROCmSoftwarePlatform/rocSOLVER)
            * `sudo apt install rocsolver` failed like rocblas
            * cloned rocsolver, building dependencies with cmake: `./install.sh -i --rocm --rocblas_dir /home/jerome/source/rocBLAS/build/release/rocblas-install`
            * needs `rocsparse`
                * `apt` fails, cloned and ran `./install.sh -d`
                * needs [rocprim](https://github.com/ROCmSoftwarePlatform/rocPRIM)
                    * cloned, ran `./install -i`

# Escaping scuffed `apt` situation

* ROCm packages were broken, couldn't install hipblas via apt
* See the second answer to [this question](https://askubuntu.com/questions/1062171/dpkg-deb-error-paste-subprocess-was-killed-by-signal-broken-pipe)

> What is happening (at least in my case) is that some renamings or other changes in library files have to be enforced because dpkg do not recognize them as belonging to the same package/program. I guess something like this happened during the installation of the nvidia drivers.

* I suspect there were name changes in the ROCm lib that were messing with apt
* Inspecting `/var/cache/apt/archive` I found dupelicates of each rocm package, with slightly different names, like:
    * `rocm-core_5.4.3.50403-121~20.04_amd64.deb`
    * `rocm-core5.4.3_5.4.3.50403-121~20.04_amd64.deb`
* So I suspect this was my issue too.
* When trying to install `hipblas` with `sudo apt install hipblas` I was getting a cryptic "configuration" related issue (sorry the actual error is lost and I can't reproduce)
* But it suggested running `sudo apt --fix-broken install`
* When running that, I was getting errors similar to (just like the stack overflow question):

```log
 trying to overwrite '/lib/udev/rules.d/71-nvidia.rules', which is also in package nvidia-kernel-common-396 396.45-0ubuntu0~gpu18.04.2
dpkg-deb: error: paste subprocess was killed by signal (Broken pipe)
```

* at the end of the log I got something like this:

```log
Errors were encountered while processing:
 /tmp/apt-dpkg-install-5PdV2S/0-rocm-core_5.4.3.50403-121~20.04_amd64.deb
 /tmp/apt-dpkg-install-5PdV2S/1-hsa-rocr_1.7.0.50403-121~20.04_amd64.deb
 /tmp/apt-dpkg-install-5PdV2S/2-hsakmt-roct-dev_20221020.0.2.50403-121~20.04_amd64.deb
 /tmp/apt-dpkg-install-5PdV2S/3-hsa-rocr-dev_1.7.0.50403-121~20.04_amd64.deb
 /tmp/apt-dpkg-install-5PdV2S/4-rocminfo_1.0.0.50403-121~20.04_amd64.deb
 /tmp/apt-dpkg-install-5PdV2S/5-comgr_2.4.0.50403-121~20.04_amd64.deb
 /tmp/apt-dpkg-install-5PdV2S/6-rocm-llvm_15.0.0.23045.50403-121~20.04_amd64.deb
 /tmp/apt-dpkg-install-5PdV2S/7-hip-runtime-amd_5.4.22804.50403-121~20.04_amd64.deb
```

* So I ended up finding the equivalent packages in `/var/cache/apt/archives` and installing them with `dpkg` with `--force-overwrite` like this (from the `/var/cache/apt/archive` folder):

```sh
 sudo dpkg -i --force-overwrite rocm-core_5.4.3.50403-121~20.04_amd64.deb
 sudo dpkg -i --force-overwrite hsa-rocr_1.7.0.50403-121~20.04_amd64.deb
 sudo dpkg -i --force-overwrite hsakmt-roct-dev_20221020.0.2.50403-121~20.04_amd64.deb
 sudo dpkg -i --force-overwrite hsa-rocr-dev_1.7.0.50403-121~20.04_amd64.deb
 sudo dpkg -i --force-overwrite rocminfo_1.0.0.50403-121~20.04_amd64.deb
 sudo dpkg -i --force-overwrite comgr_2.4.0.50403-121~20.04_amd64.deb
 sudo dpkg -i --force-overwrite rocm-llvm_15.0.0.23045.50403-121~20.04_amd64.deb
 sudo dpkg -i --force-overwrite hip-runtime-amd_5.4.22804.50403-121~20.04_amd64.deb
 sudo dpkg -i --force-overwrite rocm-llvm_15.0.0.23045.50403-121~20.04_amd64.deb
 sudo dpkg -i --force-overwrite hip-runtime-amd_5.4.22804.50403-121~20.04_amd64.deb
```

* I had to do `rocm-llvm` twice for some reason
* After that I was able to run `sudo apt --fix-broken install` successfully, then install `hipblas`
