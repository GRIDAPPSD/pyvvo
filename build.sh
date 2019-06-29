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
gld_branch=release/RC4.1

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
# NOTE: mscc: MySQL Connector/C
mscc_version="6.1.11"
mscc_dir="mysql-connector-c-${mscc_version}-linux-glibc2.12-x86_64"
mscc_archive="${mscc_dir}.tar.gz"
mscc_path="${build_dir_base}/${mscc_archive}"

# NOTE: gld --> GridLAB-D
gld=gridlab-d
gld_dir="${build_dir_base}/${gld}"
gld_tar_gz="${gld}.tar.gz"
gld_archive="${build_dir_base}/${gld_tar_gz}"

# Download mscc if it isn't already present
if [ ! -f "${mscc_path}" ]; then
    printf "Downloading MySQL Connector/C.\n\n"
    wget -P "${build_dir_base}" --no-clobber "https://dev.mysql.com/get/Downloads/Connector-C/${mscc_archive}"
fi

# If our GridLAB-D archive isn't present, clone and archive.
if [ ! -f "${gld_archive}" ]; then
    printf "Getting the latest GridLAB-D (${gld_branch} branch).\n\n"
    git clone https://github.com/gridlab-d/gridlab-d.git --branch ${gld_branch} --depth 1 --single-branch "${gld_dir}"
    printf "Compressing GridLAB-D clone."
    cd ${build_dir_base}
    tar -zcf "${gld_tar_gz}" "${gld}"
    rm -rf ${gld}
    cd ../
fi

# Move the Dockerfile.
cp Dockerfile-base ${build_dir_base}/Dockerfile

# Move into the build directory (this helps keep the Docker context
# minimal.
cd ${build_dir_base}

# Build the base container (which contains GridLAB-D, etc.)
printf "Building pyvvo-base container...\n"
docker build -t pyvvo-base:${tag} \
    --build-arg MSCC=${mscc_archive} \
    --build-arg GLD=${gld_tar_gz} \
    --build-arg MSCC_DIR_NAME=${mscc_dir} .

# Move back up.
cd ${pwd}

# Package up the PyVVO application. NOTE: This will need updated if new
# necessary folders get added.
stuff="pyvvo tests README.md"
PYVVO_ARCHIVE=pyvvo.tar.gz
tar --exclude='*__pycache__' --exclude="*.log" --exclude='*.pyc' --exclude='.git*' -zcf ${build_dir_pyvvo}/${PYVVO_ARCHIVE} ${stuff}

# Move Dockerfile and requirements.txt into pyvvo build dir.
cp Dockerfile ${build_dir_pyvvo}/Dockerfile
cp requirements.txt ${build_dir_pyvvo}/requirements.txt

# Now move into the PyVVO build directory.
cd ${build_dir_pyvvo}

# Build PyVVO
printf "Building pyvvo container...\n"
docker build -t gridappsd/pyvvo:${tag} \
    --build-arg TAG=${tag} \
    --build-arg PYVVO_ARCHIVE=${PYVVO_ARCHIVE} .
