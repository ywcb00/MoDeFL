#!/usr/bin/env bash

BASEPATH=$(dirname "$0")

ADDR_FILE="./addr.txt"
# ADJ_FILE="./adj.txt"
PROPAGATE_ARGS=()

for arg in "$@"; do # extract the address file and the adcacency matrix file from the call arguments
    case "$arg" in
        --addr_file=*) ADDR_FILE="${arg#*=}" ;;
        # --adj_file=*) ADJ_FILE="${arg#*=}" ;;
        *) PROPAGATE_ARGS+=("$arg") ;;
    esac
done

script_pid=""
trap 'echo "Killing actors and initiator."; kill -SIGTERM ${script_pid}; wait ${script_pid}' SIGINT SIGTERM

# start the actors on localhost
$BASEPATH/startLocalActors.sh "$ADDR_FILE" ${PROPAGATE_ARGS[*]} &
script_pid="$!"

# give the actors some time for startup
sleep 3

# start the initiator and the initialization phase
$BASEPATH/initiate.sh --addr_file="$ADDR_FILE" ${PROPAGATE_ARGS[*]}

wait $script_pid
