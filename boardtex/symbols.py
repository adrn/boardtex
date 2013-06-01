# coding: utf-8

from __future__ import division, print_function

# Standard library
import os, sys

_symbol_list_filename = "data/latex_symbol_list.txt"
_f = open(_symbol_list_filename)

symbols = [x.lstrip('\\').rstrip() for x in _f.readlines()]
symbol_to_idx = dict(zip(symbols,range(len(symbols))))