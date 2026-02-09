# Modular Decentralized Federated Learning Framework (MoDeFL)

### Getting Started

##### Setup
Pre-requisites
- [Git 2.43.0](https://git-scm.com/)
- [Python 3.12.3](https://www.python.org/)
- [Poetry 2.3.2](https://python-poetry.org/)
- [GNU Make 4.3](https://www.gnu.org/software/make/)

Install submodules and dependencies, and build protocol buffers
```bash
make install
```

Check if everything was installed successfully by running
```bash
make sanitycheck
```

##### Deploy
Deploy an actor via the following command
```bash
make act PORT=<port>
```

Deploy the initiator via the following command
```bash
make initiate ADDR_FILE=<addr_file> ADJ_FILE=<adj_file>
```
