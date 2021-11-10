#!/bin/bash
sourcefile=/opt/gfa/python
[[ -e $sourcefile ]] && . $sourcefile 3.7
[[ $(hostname) == "pc11292.psi.ch" ]] && export PYTHONPATH=$PYTHONPATH:~/pythonpath2/pyqt5
python3 main.py
