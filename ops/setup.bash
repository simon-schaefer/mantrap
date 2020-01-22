#!/bin/bash

# Setup environment variables.
PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VIRTUAL_ENV=".venv_mantrap"
echo 'Setting up project ...'

# Login to virtual environment.
cd "${PROJECT_HOME}" || return
if [[ ! -d "${VIRTUAL_ENV}" ]]; then
    echo 'Creating virtual environment ...'
    mkdir "${VIRTUAL_ENV}"
    pip3 install virtualenv
    virtualenv -p python3 "${VIRTUAL_ENV}"
fi
# shellcheck source=.venv_muresco/bin/activate
source "${PROJECT_HOME}"/"${VIRTUAL_ENV}"/bin/activate

# Install self-python-package.
echo $'\nInstalling package ...'
cd "${PROJECT_HOME}" || return
pip3 install -r "${PROJECT_HOME}"/ops/requirements.txt
cp "${PROJECT_HOME}"/ops/setup.py "${PROJECT_HOME}"
pip3 install -e .
rm "${PROJECT_HOME}"/setup.py

cd "${PROJECT_HOME}" || return
echo $'\nSuccessfully set up project !'