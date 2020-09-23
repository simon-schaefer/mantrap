#!/bin/bash

# Setup environment variables.
PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."
EXTERNAL_HOME="$PROJECT_HOME/third_party"
VIRTUAL_ENV=".venv_mantrap"
INSTALL_SGAN=true

echo '==> Setting up project ...'

# Install package requirements.
echo $'==> Installing package requirements ...'
cd "${PROJECT_HOME}" || return
pip3 install -r "${PROJECT_HOME}"/ops/requirements.txt
if [ "$INSTALL_SGAN" = true ]; then
  pip3 install -r "${EXTERNAL_HOME}"/sgan/requirements.txt
fi
pip3 install -r "${EXTERNAL_HOME}"/GenTrajectron/requirements.txt

# Install project packages.
echo $'==> Installing project packages ...'
cp "${PROJECT_HOME}"/ops/setup.py "${PROJECT_HOME}"
pip3 install -e . --quiet
rm "${PROJECT_HOME}"/setup.py

# Download sgan model.
if [ "$INSTALL_SGAN" = true ]; then
  cd "${EXTERNAL_HOME}"/sgan && bash scripts/download_models.sh
fi

# Create output directory.
echo $'==> Building project structure ...'
mkdir -p "${PROJECT_HOME}"/outputs

cd "${PROJECT_HOME}" || return
echo $'\n==> Successfully set up project !'
