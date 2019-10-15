#!/bin/bash

PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/..
SCENARIO_LABEL="$(openssl rand -hex 3)"

python3 "$PROJECT_HOME"/scripts/build_env.py "$SCENARIO_LABEL"
convert -delay 30 "$PROJECT_HOME"/config/"$SCENARIO_LABEL"/*.png "$PROJECT_HOME"/config/"$SCENARIO_LABEL"/scene.mpg
