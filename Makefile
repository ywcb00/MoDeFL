
install: update-submodules install-dependencies build-protocolbuffers

update-submodules:
	@echo "Updating submodules"
	git submodule init
	git submodule update

install-dependencies:
	@echo "Installing Poetry dependencies"
	poetry install

build-protocolbuffers:
	@echo "Compiling Protocol Buffers"
	poetry run python -m grpc_tools.protoc -Inetwork/protos=./network/protobuf --python_out=./ --pyi_out=./ --grpc_python_out=./ ./network/protobuf/*.proto

# call like `make initiate ADDR_FILE=<addr_file> ADJ_FILE=<adj_file> [CONFIG_FILE=<config_file>]`
initiate:
	@echo "Starting the Initiator"
	poetry run python main.py --initiate \
		--config=$(CONFIG_FILE) \
		--addr_file=$(ADDR_FILE) \
		--adj_file=$(ADJ_FILE)

# call like `make act PORT=<port> [CONFIG_FILE=<config_file>]`
act:
	@echo "Starting the Actor"
	poetry run python main.py --act \
		--config=$(CONFIG_FILE) \
		--port=$(PORT)
	echo "After actor"

sanitycheck:
	@make act PORT=50506 \
		CONFIG_FILE="./resources/config/config_sanitycheck.json" > /dev/null &
	@sleep 3 && make initiate \
		ADDR_FILE="./resources/addr_sanitycheck.txt" \
		ADJ_FILE="./resources/adj_sanitycheck.txt" > /dev/null \
		CONFIG_FILE="./resources/config/config_sanitycheck.json" &
	@make act PORT=50505 \
		CONFIG_FILE="./resources/config/config_sanitycheck.json" > /dev/null
	@echo "Sanity seems fine."
