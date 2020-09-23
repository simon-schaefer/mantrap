#!/bin/bash

IPOPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Download
# Install Ipopt NLP solver.
cd "${IPOPT_PATH}" || return
chmod u+x coinbrew
brew install bash  # update bash version (>= 4.0)
brew install pkg-config

echo ">>Building IPOPT"
mkdir build
./coinbrew build Ipopt --prefix="${IPOPT_PATH}/build" --test --no-prompt
./coinbrew install Ipopt

# Set PKG_CONFIG_PATH environment variable to IPOPT build directory
export PKG_CONFIG_PATH="${IPOPT_PATH}/build/Ipopt/master"
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:${IPOPT_PATH}/build/ThirdParty/Mumps/2.1"
export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:${IPOPT_PATH}/build/ThirdParty/Metis/2.0"
export DYLD_LIBRARY_PATH="${IPOPT_PATH}/build/lib"

echo ">>PKG_CONFIG_PATH: "
echo "${PKG_CONFIG_PATH}" || return

# Install cyipopt following https://pypi.org/project/ipopt/
# Download binary files from https://pypi.org/project/ipopt/#files
# Then install by running
cd ..

echo ">>Building cyipopt"
wget https://files.pythonhosted.org/packages/05/57/a7c5a86a8f899c5c109f30b8cdb278b64c43bd2ea04172cbfed721a98fac/ipopt-0.1.9.tar.gz
tar -xzvf ipopt-0.1.9.tar.gz
rm ipopt-0.1.9.tar.gz

mv ipopt-0.1.9 cyipopt
cd cyipopt || return
python3 setup.py install
