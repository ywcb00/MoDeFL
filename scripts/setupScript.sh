#!/usr/bin/env bash

BASEPATH=$(dirname "$0")
ROOTPATH="$BASEPATH/.."

python -m grpc_tools.protoc -Inetwork/protos=$ROOTPATH/network/protobuf --python_out=$ROOTPATH --pyi_out=$ROOTPATH --grpc_python_out=$ROOTPATH $ROOTPATH/network/protobuf/*.proto