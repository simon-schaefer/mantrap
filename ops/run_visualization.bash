#!/bin/bash

PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/..
# shellcheck source=.venv_mantrap/bin/activate
source "${PROJECT_HOME}"/.venv_mantrap/bin/activate

rm -r "${PROJECT_HOME}"/test/graphs
mkdir "${PROJECT_HOME}"/test/graphs
python3 "${PROJECT_HOME}"/test/visualize.py

cd "${PROJECT_HOME}"/test/graphs || return
for d in */ ; do
  cd "${d}" || return
  ffmpeg -loglevel panic -r 10 -i %04d.png -vcodec mpeg4 -y movie.mp4
  cd "${PROJECT_HOME}"/test/graphs || return
done