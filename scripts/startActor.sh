#!/usr/bin/env bash

BASEPATH=$(dirname "$0")
ROOTPATH="$BASEPATH/.."

ADDR="localhost:50051"

PROPAGATE_ARGS=()

for arg in "$@"; do # extract the address file from the call arguments
    case "$arg" in
        --address=*) ADDR="${arg#*=}" ;;
        *) PROPAGATE_ARGS+=("$arg") ;;
    esac
done

PORT=$(echo $ADDR | cut -d : -f 2)
# start the main python script with the actor flag
python $ROOTPATH/main.py --act --port=$PORT ${PROPAGATE_ARGS[*]}
