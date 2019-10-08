#!/bin/bash

PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/..
# shellcheck source=venv/bin/activate
source "${PROJECT_HOME}"/.venv/bin/activate
pytest "${PROJECT_HOME}"/test/*.py -vv