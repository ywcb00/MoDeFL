#!/usr/bin/env bash

python -m grpc_tools.protoc -Inetwork/protos=./network/protobuf --python_out=. --pyi_out=. --grpc_python_out=. ./network/protobuf/*.proto