#!/usr/bin/env bash
# Use this to build pyvvo.
#
# Prereqs: git, wget
#
# Arguments:
#   - tag: tag for image. Image name will be pyvvo:<tag>. Defaults to
#          "latest"

# Grab this directory (though we shouldn't need to...)
pwd=$PWD

# Constants:
# NOTE: When updating this branch, you probably should delete the
# GridLAB-D directory ($gld_dir) first - the behavior of using
# --single-branch with git clone does some weird stuff.
gld_branch=release/v4.2

# Collect arguments.
tag=${1:-latest}

# Assign image name
image="pyvvo:${tag}"
printf "Docker image will be named ${image}.\n\n"

# Define some constants.
build_dir_base="build/base"
build_dir_pyvvo="build/pyvvo"
mkdir -p ${build_dir_base}
mkdir -p ${build_dir_pyvvo}
# We'll be setting up the MySQL apt repository.
mysql_apt="mysql-apt-config_0.8.13-1_all.deb"
mysql_apt_path="${build_dir_base}/${mysql_apt}"
# To run dpkg -i on the $mysql_apt file, we need to install the
# following dependencies:
mysql_apt_deps="lsb-release wget gnupg"

# NOTE: gld --> GridLAB-D
gld=gridlab-d
gld_dir="${build_dir_base}/${gld}"
gld_tar_gz="${gld}.tar.gz"
gld_archive="${build_dir_base}/${gld_tar_gz}"

# Download mysql apt configuration if it isn't already present
if [ ! -f "${mysql_apt_path}" ]; then
    printf "Downloading mysql-apt-config.\n\n"
    wget -P "${build_dir_base}" --no-clobber "https://dev.mysql.com/get//${mysql_apt}"
fi

# If our GridLAB-D archive isn't present, clone and archive.
if [ ! -f "${gld_archive}" ]; then
    printf "Getting the latest GridLAB-D (${gld_branch} branch).\n\n"
    git clone https://github.com/gridlab-d/gridlab-d.git --branch ${gld_branch} --depth 1 --single-branch "${gld_dir}"
    printf "Compressing GridLAB-D clone."
    cd ${build_dir_base}
    tar -zcf "${gld_tar_gz}" "${gld}"
    rm -rf ${gld}
    cd ${pwd}
fi

# Move the Dockerfile.
cp Dockerfile-base ${build_dir_base}/Dockerfile
# Move the helper script.
cp install_libmysqlclient-dev.sh ${build_dir_base}/install_libmysqlclient-dev.sh

# Move into the build directory (this helps keep the Docker context
# minimal.
cd ${build_dir_base}

# Build the base container (which contains GridLAB-D, etc.)
printf "Building pyvvo-base container...\n"
docker build -t pyvvo-base:${tag} \
    --build-arg mysql_apt=${mysql_apt} \
    --build-arg mysql_apt_deps="${mysql_apt_deps}" \
    --no-cache \
    --build-arg GLD=${gld_tar_gz} .

# Move back up.
cd ${pwd}

# Package up the PyVVO application. NOTE: update packaging.txt to add
# files/directories.
stuff=$(cat packaging.txt | egrep -v "^\s*(#|$)")
PYVVO_ARCHIVE=pyvvo.tar.gz
tar --exclude='*__pycache__' --exclude="*.log" --exclude='*.pyc' --exclude='.git*' -zcf ${build_dir_pyvvo}/${PYVVO_ARCHIVE} ${stuff}

# Move files into pyvvo build dir.
cp Dockerfile ${build_dir_pyvvo}/Dockerfile
cp ${mysql_apt_path} ${build_dir_pyvvo}/${mysql_apt}
cp install_libmysqlclient-dev.sh ${build_dir_pyvvo}/install_libmysqlclient-dev.sh
cp requirements.txt ${build_dir_pyvvo}/requirements.txt
cp setup.py ${build_dir_pyvvo}/setup.py

# Now move into the PyVVO build directory.
cd ${build_dir_pyvvo}

# Build PyVVO
printf "Building pyvvo container...\n"
docker build -t gridappsd/pyvvo:${tag} \
    --build-arg TAG=${tag} \
    --build-arg PYVVO_ARCHIVE=${PYVVO_ARCHIVE} \
    --build-arg mysql_apt=${mysql_apt} \
    --no-cache \
    --build-arg mysql_apt_deps="${mysql_apt_deps}" .
