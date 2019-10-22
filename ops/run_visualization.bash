#!/bin/bash

PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/..
# shellcheck source=.venv_muresco/bin/activate
source "${PROJECT_HOME}"/.venv_muresco/bin/activate

rm -r "${PROJECT_HOME}"/test/graphs
mkdir "${PROJECT_HOME}"/test/graphs
python3 "${PROJECT_HOME}"/test/visualize.py

cd "${PROJECT_HOME}"/test/graphs || return
for d in */ ; do
  convert -delay 30 "$PROJECT_HOME"/test/graphs/"$d"*.png "$PROJECT_HOME"/test/graphs/"$d"scene.mpg
done