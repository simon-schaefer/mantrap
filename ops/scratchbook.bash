#!/bin/bash

PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/..
PROJECT_NAME="$( dirname "${PROJECT_HOME}" )"
# shellcheck source=venv/bin/activate
source "${PROJECT_HOME}"/.venv/bin/activate
ipython kernel install --user --name="${PROJECT_NAME}"
jupyter notebook "$PROJECT_HOME"/notebooks/scratchbook.ipynb
