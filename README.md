# GPT Learning repository

A collection of tutorials from various places, or exercises recommended by other people / ChatGPT, some little projects here and there.

# Deep Learning "Hello World" Projects

Following are a few "Hello World" like machine learning projects suggested by ChatGPT itself.

## Image classification

Build a deep neural network using PyTorch to classify images in the CIFAR-10 dataset. This is a classic "hello world" project for deep learning and will help you get comfortable with building and training neural networks.

* Implemented here: [conv_net](/tut/conv_net.ipynb)

### Next Steps

Try and prevent overfitting with these techniques:

* L1 and L2 regularization: These are techniques that add a penalty term to the loss function to discourage the model from assigning too much importance to any single feature. L1 regularization adds the absolute value of the weights to the loss function, while L2 regularization adds the square of the weights. L2 regularization is also known as weight decay.
* Dropout: This technique randomly drops out some of the neurons in a layer during training, which forces the remaining neurons to take on more responsibility and prevents overfitting.
* Early stopping: This technique stops training the model before it fully converges to the training data, based on a validation set's performance. It helps prevent overfitting by selecting a model with the best generalization performance, rather than the one that fits the training data the best.
* Data augmentation: This technique artificially increases the size of the training set by applying transformations, such as rotations or flips, to the input data. This helps the model learn more robust features and prevents overfitting.
* Batch normalization: This technique normalizes the inputs to a layer across each mini-batch of data during training. It helps reduce the internal covariate shift problem and regularizes the model by making it less sensitive to the scale of the inputs.

## Sentiment analysis

Use PyTorch to build a model that can predict the sentiment of a text message. You can use a dataset like IMDB reviews or Twitter sentiment analysis dataset.

### Sub-Tasks

* Exploring different NLP pre-processing techniques: [1_1_pre_processing_learning](tut/sentiment_analysis/1_1_pre_processing_learning.ipynb)
* Training a tokenizer with BPE: [1_2_tokenizer](tut/sentiment_analysis/1_2_tokenizer.ipynb)
* Sentiment analysis with an RNN: [2_1_rnn](tut/sentiment_analysis/2_1_rnn.ipynb)
    * Implemented using `nn.RNN` easily
    * Implemented custom RNN module, however had issues training it due to instability (gradient exploding, and then vanishing). Deferred learning how to fix this to a later date

* Sentiment analysis with an LSTM: [2_2_lstm](tut/sentiment_analysis/2_2_lstm.ipynb)
* Sentiment analysis with a GRU: [2_3_gru](tut/sentiment_analysis/2_3_gru.ipynb)
* Sentiment analysis with a CNN: [2_4_cnn](tut/sentiment_analysis/2_4_cnn.ipynb)
* Sentiment analysis with a transformer: [2_5_transformer](tut/sentiment_analysis/2_5_transformer.ipynb)
* Sentiment analysis with pre-trained models:
    * [bert](tut/sentiment_analysis/2_6_pretrained_bert.ipynb)
    * [gpt](tut/sentiment_analysis/2_7_pretrained_gpt.ipynb)
    * [roberta](tut/sentiment_analysis/2_8_pretrained_roberta.ipynb)

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
