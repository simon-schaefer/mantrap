# /bin/bash

DOCKER_NAME="seleschaefer"
PACKAGE="mantrap"
TAG="0.1"

docker login
docker build --rm -t $DOCKER_NAME/$PACKAGE:$TAG .
docker push $DOCKER_NAME/$PACKAGE:$TAG
