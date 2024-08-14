#!/usr/bin/env bash

BASEPATH=$(dirname "$0")
SCRIPTPATH="$BASEPATH/.."
ROOTPATH="$BASEPATH/../.."


PROPAGATE_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --log_dir=*) ARG_LOGDIR="${arg#*=}" ;;
        --num_epochs=*) ARG_NUM_EPOCHS="${arg#*=}" ;;
        *) PROPAGATE_ARGS+=("$arg") ;;
    esac
done


trap 'echo "Interrupting script and exiting."; exit 0;' SIGINT SIGTERM

client_lr=(0.1 0.05 0.02 0.01 0.005 0.002)
NUM_EPOCHS=${ARG_NUM_EPOCHS:-10}

for clr in "${client_lr[@]}"
do
    LOGDIR="${ARG_LOGDIR:-"$ROOTPATH/log"}/clr$clr"
    echo "=== Performing experiment with client learning rate $clr. ==="
    $SCRIPTPATH/startAndInitiate.sh --lr_client="$clr" --log_dir="$LOGDIR" --num_epochs=$NUM_EPOCHS ${PROPAGATE_ARGS[*]}
done
