#!/usr/bin/env bash

BASEPATH=$(dirname "$0")
ROOTPATH="$BASEPATH/.."

ADDR="localhost:50051"

PROPAGATE_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --address=*) ADDR="${arg#*=}" ;;
        *) PROPAGATE_ARGS+=("$arg") ;;
    esac
done

PORT=$(echo $ADDR | cut -d : -f 2)
python $ROOTPATH/main.py --act --port=$PORT ${PROPAGATE_ARGS[*]}
