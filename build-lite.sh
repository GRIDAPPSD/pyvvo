#!/usr/bin/env bash
# Use this to build the "lite" version of PyVVO. GridLAB-D (and
# associated dependencies) won't be installed. Additionally, PyVVO will
# be pip installed.
#
# Arguments:
#   - tag: tag for image. Note that "-lite" will be appended. Defaults
#       to "latest"

# Setup tag and image name.
tag=${1:-latest}-lite
image="gridappsd/pyvvo:${tag}"

# Create directory for building.
build_dir=build/lite
mkdir -p ${build_dir}

# Move files into pyvvo build dir.
cp Dockerfile-lite ${build_dir}/Dockerfile
cp requirements.txt ${build_dir}/requirements.txt
# We're going to remove the 'mysqlclient' dependency here.
sed "s@'mysqlclient',@@" setup.py > ${build_dir}/setup.py

# Package up the PyVVO application.
# https://kvz.io/blog/2007/07/11/cat-a-file-without-the-comments/
to_package=$(cat packaging.txt | egrep -v "^\s*(#|$)")
archive=pyvvo.tar.gz
tar --exclude='*__pycache__' --exclude="*.log" --exclude='*.pyc' --exclude='.git*' -zcf ${build_dir}/${archive} ${to_package}

# Move into the build directory.
cd ${build_dir}

# Build.
docker build -t ${image} \
    --build-arg PYVVO_ARCHIVE=${archive} .
