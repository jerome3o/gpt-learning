# PyTorch Learning Repo

A collection of tutorials from various places, or exercises recommended by other people / ChatGPT

# PyTorch "Hello World" Projects

Following are a few "Hello World" like machine learning projects suggested by ChatGPT itself.

## Image classification

Build a deep neural network using PyTorch to classify images in the CIFAR-10 dataset. This is a classic "hello world" project for deep learning and will help you get comfortable with building and training neural networks.

* See [conv_net](/tut/conv_net.ipynb)

## Sentiment analysis

Use PyTorch to build a model that can predict the sentiment of a text message. You can use a dataset like IMDB reviews or Twitter sentiment analysis dataset.

## Handwritten digit recognition

Build a deep neural network using PyTorch to recognize handwritten digits. You can use the MNIST dataset for this project, which contains 60,000 training images and 10,000 testing images of handwritten digits.

## Generative Adversarial Networks (GANs)

Build a simple GAN using PyTorch to generate images of faces or objects. This is a great project to get an introduction to the concepts of generative modeling.

## Reinforcement learning

Use PyTorch to build a reinforcement learning agent that can learn to play a simple game like CartPole. This will help you get comfortable with building and training agents using RL algorithms.

# Setup

## Python environment setup

You'll need python 3.8+ (whatever works with torch), pip, and venv

Create and activate venv

```sh
python -m venv venv
./venv/bin/activate
``` 

Install reqs
```
pip install -r requirements.txt
```

## Jupyter Server

### Initial setup

Here are some helpers for setting up a jupyterlab server with a password

Scaffold config file:
```sh
jupyter notebook --generate-config
```

Generate a password:
```
python -c 'from notebook.auth import passwd; print(passwd())'
```

Copy the stdout from about and add it to the file: `~/.jupyter/jupyter_notebook_config.py` with:
```python
c.NotebookApp.password = 'PASTE-HASHED-PW-FROM-BEFORE'
```

Some other useful settings:
```python
# if you're running it on a server
c.NotebookApp.open_browser = False

# if you'd like to run it on a specific address, or bind to all.
# I bound it to it's tailscale address so I can access only from 
#   the vpn
c.NotebookApp.ip = '0.0.0.0'
```

I've also made a systemd service unit so it will start up on launch (could probably do it in docker but the ROCm image is huge):
```ini
# /etc/systemd/system/jupyter.service

[Unit]
Description=Jupyter server for tinking with machine learning
; After=network.target

[Service]
User=jerome
WorkingDirectory=/home/jerome/source/pytorch_hello_worlds/
ExecStart=/home/jerome/source/pytorch_hello_worlds/start_jupyter.sh
Restart=always

[Install]
WantedBy=multi-user.target
```
