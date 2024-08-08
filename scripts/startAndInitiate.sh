#!/usr/bin/env bash

BASEPATH=$(dirname "$0")

ADDR_FILE="resources/actor_addresses.txt"
ADJ_FILE="./resources/actor_adjacency.txt"
PROPAGATE_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --addr_file=*) ADDR_FILE="${arg#*=}" ;;
        --adj_file=*) ADJ_FILE="${arg#*=}" ;;
        *) PROPAGATE_ARGS+=("$arg") ;;
    esac
done

script_pid=()

$BASEPATH/startLocalActors.sh "$ADDR_FILE" "${PROPAGATE_ARGS[*]}" &
script_pid+=("$!")

sleep 3

$BASEPATH/initiate.sh --addr_file="$ADDR_FILE" --adj_file="$ADJ_FILE" "${PROPAGATE_ARGS[*]}" &
script_pid+=("$!")

wait ${script_pid[*]}
