# This is a Makefile that can be used to create a Docker image and
# running a container ready for using EstimationPy.
#
# NOTE:
# Before starting to build the Docker image make sure that you have
# downloaded Sundials version 2.6.2 from https://computation.llnl.gov/casc/sundials/main.html
# and that the tar.gz file is located in the dockers/softwares folder
#
IMG_NAME=lbl-srg/estimationpy_box
CONTAINER_NAME=estimationpy_box

all: build_image

build_image:
	docker build -t=${IMG_NAME} --rm=true .

remove_image:
	docker rmi ${IMG_NAME}

start_container_bash:
	docker run -t -i \
	-p 8888:8888 \
	-v ${CURDIR}/shared_folder:/home/docker/shared_folder \
	${IMG_NAME} /bin/bash

start_container_ipynb:
	docker run -d \
	-p 8888:8888 \
	-v ${CURDIR}/shared_folder:/home/docker/shared_folder \
	${IMG_NAME}

stop_container:
	docker stop ${IMG_NAME}
