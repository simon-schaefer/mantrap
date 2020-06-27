#!/bin/bash

# Setup environment variables.
PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
EXTERNAL_HOME="$PROJECT_HOME/third_party"
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

# Install package requirements.
echo $'Installing package requirements ...'
cd "${PROJECT_HOME}" || return
pip3 install -r "${PROJECT_HOME}"/ops/requirements.txt --quiet
pip3 install -r "${EXTERNAL_HOME}"/sgan/requirements.txt --quiet
pip3 install -r "${EXTERNAL_HOME}"/GenTrajectron/requirements.txt --quiet

# Install project packages.
echo $'Installing project packages ...'
cp "${PROJECT_HOME}"/ops/setup.py "${PROJECT_HOME}"
pip3 install -e . --quiet
rm "${PROJECT_HOME}"/setup.py

# Create output directory.
echo $'Building project structure ...'
mkdir -p "${PROJECT_HOME}"/outputs

cd "${PROJECT_HOME}" || return
echo $'\nSuccessfully set up project !'
