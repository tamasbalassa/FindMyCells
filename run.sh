#!/bin/bash

DIR=$(dirname "$(readlink -f "$0")")
FMCPATH=$DIR/AI

export PYTHONPATH=$PYTHONPATH:$FMCPATH

python3 graphics/main_window.py