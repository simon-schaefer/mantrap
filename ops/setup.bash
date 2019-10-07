#!/bin/bash

# Setup environment variables.
PROJECT_OPS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_HOME="$PROJECT_OPS/.."
VIRTUAL_ENV=".venv_muresco"
echo 'Setting up project ...'

# Login to virtual environment.
cd "${PROJECT_HOME}" || return
if [[ ! -d "${VIRTUAL_ENV}" ]]; then
    echo 'Creating virtual environment ...'
    mkdir "${VIRTUAL_ENV}"
    virtualenv -p python3 "${VIRTUAL_ENV}"
fi
# shellcheck source=venv/bin/activate
source "${PROJECT_HOME}"/"${VIRTUAL_ENV}"/bin/activate

# Install self-python-package.
echo $'\nInstalling package ...'
cd "${PROJECT_HOME}" || return
pip3 install -r ops/requirements.txt
cp "${PROJECT_OPS}"/setup.py "${PROJECT_HOME}"
pip3 install -e .
rm "${PROJECT_HOME}"/setup.py

cd "${PROJECT_HOME}" || return
echo $'\nSuccessfully set up project !'
