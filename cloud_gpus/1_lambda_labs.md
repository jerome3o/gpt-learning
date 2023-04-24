# Lambda Labs GPU setup

* A10 GPU
* Ubuntu amd64 20.04

## Setup

Make sure agent forwarding is configured in your ssh config i.e.

```config
Host agentc
  HostName {IP}
  User ubuntu
  ForwardAgent yes
```

* Update packages
    * `sudo apt update`
    * `sudo apt upgrade`
* Installing python 3.10
    * `sudo apt install python3.10 python3.10-dev python3.10-venv`
* Make that the main python (hack warning)
    * `sudo cp /usr/bin/python3.10 /usr/bin/python3`
    * `sudo cp /usr/bin/python3.10 /usr/bin/python`
* Install dotfiles (for comfort if developing)
    * `git clone git@github.com:jerome3o/dotfiles`
    * `cd dotfiles`
    * `sudo apt install stow`
    * `./stow_all.sh`
    * `cd`
    * `./scripts/terminal.sh`
* Configure your project, usually you will need `tranformers`, `accelerate`, and `bitsandbytes` from pip to run quantized models
* There was an issue with bitsandbytes, [this](https://github.com/TimDettmers/bitsandbytes/issues/156#issuecomment-1462329713) was helpful.
