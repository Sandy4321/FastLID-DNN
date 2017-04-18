#!/bin/python 

import sys

if len(sys.argv) != 3:
    print "Usage: python ndx2flist.py <ndx filename> <flist filename>"
    sys.exit(1)

ndx = open(sys.argv[1], 'r')
flist = open(sys.argv[2], 'w')
