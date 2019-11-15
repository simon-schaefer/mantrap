#!/bin/bash

# Setup environment variables.
PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo 'Setting up project ...'



cd "${PROJECT_HOME}" || return
echo $'\nSuccessfully set up project !'
