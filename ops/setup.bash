#!/bin/bash

# Setup environment variables.
PROJECT_OPS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# PROJECT_NAME="$( dirname "${PROJECT_OPS}" )"
PROJECT_HOME="$PROJECT_OPS/.."
echo 'Setting up project ...'

# Login to virtual environment.
cd "${PROJECT_HOME}" || return
if [[ ! -d 'venv' ]]; then
    echo 'Creating virtual environment ...'
    mkdir venv
    virtualenv -p python3 venv
fi
# shellcheck source=venv/bin/activate
source "${PROJECT_HOME}"/venv/bin/activate

# Install self-python-package.
echo $'\nInstalling package ...'
cd "${PROJECT_HOME}" || return
pip3 install -r requirements.txt
pip3 install -e .

cd "${PROJECT_HOME}" || return
echo $'\nSuccessfully set up project !'
