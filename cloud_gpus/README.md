# Cloud GPU compute

With int8 matmul not working on my AMD GPUs I will see if I can get some cheap NVIDIA cards, but in the mean time I want to explore cloud options to continue learning/testing.

The goal is to rent a consumer (or relatively cheap/attainable) GPU and try get some LLMs running on it. Following are some notes on different cloud providers

## Google Cloud

They have a bunch of GPUs available, generally only in the more populous regions i.e. not sydney, but a bunch in the US.

GPUs
* T4:
    * ~$201 a month with usage discount applied

Notes
* Usage discount:
    * > Whenever you use an applicable resource for more than a fourth of a billing month, you automatically receive a discount for every incremental hour that you continue to use that resource.

## AWS

Couldn't find a way to hire a VPS with a GPU. This may be because I have the wrong type of account, or I just missed something obvious.

## Lambda Labs

Only have high end GPUs, only one available are:

```
1x A10 (24 GB PCIe)
30 vCPUs, 200 GiB RAM, 1.4 TiB SSD
$0.60 / hr
```

Seeing as this was the only provider I could easily get working, I have started looking into the setup with the A10.

See [this doc](/cloud_gpus/1_lambda_labs.md) for more details.

## Azure

Looks like there are GPUs on there, but after playing around for a bit I couldn't figure out how. You may need to set up a VM first and then add a GPU.

Appears to be support for T4's [here](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series), but I can't get to it beyond the docs.

## Digital Ocean

Looks like they don't currentl support GPUs on their VMs. [Recommendation thread "under review"](https://ideas.digitalocean.com/core-compute-platform/p/add-gpu-instances)

## Linode

They have RTX6000 GPUs available in 1,2,3,4x configurations. They are $1000,$2000,$3000,$4000 per month respectively.
