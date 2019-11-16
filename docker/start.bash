# /bin/bash

DOCKER_NAME="seleschaefer"
PACKAGE="mantrap"
TAG="0.1"

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker login
docker pull $DOCKER_NAME/$PACKAGE:$TAG
docker run -d -t --name="mantrap" -v ${SCRIPTPATH}/..:/home/catkin_ws/src/mantrap $DOCKER_NAME/$PACKAGE:$TAG
