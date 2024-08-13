#!/usr/bin/env bash

BASEPATH=$(dirname "$0")
SCRIPTPATH="$BASEPATH/.."
ROOTPATH="$BASEPATH/../.."

trap 'echo "Interrupting script and exiting."; exit 0;' SIGINT SIGTERM

client_lr=(0.1 0.05 0.02 0.01 0.005 0.002)
num_epochs=20

for clr in "${client_lr[@]}"
do
    echo "=== Performing experiment with client learning rate $clr. ==="
    $SCRIPTPATH/startAndInitiate.sh --lr_client="$clr" --log_dir="$ROOTPATH/log/gslr/lr$clr" --num_epochs=$num_epochs "$@"
done
