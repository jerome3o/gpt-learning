# PyTorch Learning Repo

A collection of tutorials from various places, or exercises recommended by other people / ChatGPT

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
