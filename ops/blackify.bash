#!/bin/bash

PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/..
OPTIONS="-l 120"

echo "Reformatting src files ..."
black "$PROJECT_HOME"/mantrap/*.py "$OPTIONS"
for dir in "$PROJECT_HOME"/mantrap/*/
do
  for file in "$dir"/*.py; do
    [ -f "$file" ] || break
    black "$file" "$OPTIONS"
  done
done

echo "Reformatting test files ..."
cd "$PROJECT_HOME"/test || return
black *.py "$OPTIONS"

echo "Reformatting scripts files ..."
cd "$PROJECT_HOME"/scripts || return
black *.py "$OPTIONS"
