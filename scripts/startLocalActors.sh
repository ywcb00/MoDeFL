#!/usr/bin/env bash

BASEPATH=$(dirname "$0")

ADDR_FILE=${1:-"resources/addr.txt"}
shift # remove the first argument from argument list (i.e., the address file)

actor_pid=() # empty array
trap 'echo "Killing local actors."; for apid in ${actor_pid[*]}; do pkill -SIGTERM -P ${apid}; done; wait ${actor_pids[*]}' SIGINT SIGTERM

IFS=$'\n'   # make newlines the only separator
set -f      # disable glob patterns
for ADDR in $(cat "$ADDR_FILE"); do
    echo $ADDR
    $BASEPATH/startActor.sh --address=$ADDR "$@" &
    actor_pid+=("$!")
done

wait ${actor_pid[*]}

exit
