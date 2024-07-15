#!/usr/bin/env bash

ADDR="localhost:50051"

for arg in "$@"; do
    case "$arg" in
        --address=*) ADDR="${arg#*=}" ;;
    esac
done

PORT=$(echo $ADDR | cut -d : -f 2)
python main.py --act --port=$PORT

exit
