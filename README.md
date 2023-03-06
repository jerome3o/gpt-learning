# PyTorch Learning Repo

A collection of tutorials from various places, or exercises recommended by other people / ChatGPT

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
ExecStart=/home/jerome/source/pytorch_hello_worlds/venv/bin/jupyter lab
Restart=always

[Install]
WantedBy=multi-user.target
```
