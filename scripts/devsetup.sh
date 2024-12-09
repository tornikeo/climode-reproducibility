#!/bin/bash
## This is setup for development
set -e

pip install -e .

# vscode stuff
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extensions ms-python.black-formatter