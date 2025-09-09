#!/usr/bin/env bash

BASEPATH=$(dirname "$0")
SCRIPTPATH="$BASEPATH/.."
ROOTPATH="$BASEPATH/../.."

trap 'echo "Interrupting script and exiting."; exit 0;' SIGINT SIGTERM


lt=1
echo "=== Performing experiment with learning type DFLv$lt. ==="
$BASEPATH/gridSearchLearningRate.sh --learning_type=$lt --log_dir="$ROOTPATH/log/dflv$lt" --addr_file="$ROOTPATH/resources/addr.txt" --adj_file="$ROOTPATH/resources/adj_dflv1.txt"

lt=2
echo "=== Performing experiment with learning type DFLv$lt. ==="
$BASEPATH/gridSearchLearningRate.sh --learning_type=$lt --log_dir="$ROOTPATH/log/dflv$lt" --addr_file="$ROOTPATH/resources/addr.txt" --adj_file="$ROOTPATH/resources/adj_dflv1.txt"

lt=3
echo "=== Performing experiment with learning type DFLv$lt. ==="
$BASEPATH/gridSearchLearningRate.sh --learning_type=$lt --log_dir="$ROOTPATH/log/dflv$lt" --addr_file="$ROOTPATH/resources/addr.txt" --adj_file="$ROOTPATH/resources/adj_dflv1.txt"
