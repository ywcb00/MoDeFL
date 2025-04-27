#!/usr/bin/env bash

BASEPATH=$(dirname "$0")
ROOTPATH="$BASEPATH/.."

# start the main python sctipt with the initiator flag
python $ROOTPATH/main.py --initiate "$@"

exit
