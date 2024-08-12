#!/usr/bin/env bash

BASEPATH=$(dirname "$0")
ROOTPATH="$BASEPATH/.."

python $ROOTPATH/main.py --initiate "$@"

exit
