#!/usr/bin/env bash
# Use this to build pyvvo.
#
# Prereqs: git, wget
#
# Arguments:
#   - tag: tag for image. Image name will be pyvvo:<tag>. Defaults to
#          "latest"

# Collect arguments.
tag=${1:-latest}

# Assign image name
image="pyvvo:${tag}"
printf "Docker image will be named ${image}.\n\n"

# Define some constants.
build_dir="build"
# NOTE: mscc: MySQL Connector/C
mscc_version="6.1.11"
mscc_dir="mysql-connector-c-${mscc_version}-linux-glibc2.12-x86_64"
mscc_archive="${mscc_dir}.tar.gz"
mscc_path="${build_dir}/${mscc_dir}"

# NOTE: gld: GridLAB-D
gld_dir="${build_dir}/gridlab-d"

# Download and extract mscc if it isn't already present
if [ ! -d "${mscc_path}" ]; then
    printf "Downloading and extracting MySQL Connector/C.\n\n"
    wget -P "${build_dir}" --no-clobber "https://dev.mysql.com/get/Downloads/Connector-C/${mscc_archive}"
    tar -zxf "${build_dir}/${mscc_archive}" --directory "${build_dir}"
fi

# Clone GridLAB-D, or pull the latest.
echo "Getting the latest GridLAB-D (develop branch).\n\n"
git clone https://github.com/gridlab-d/gridlab-d.git -b develop --single-branch "${gld_dir}" 2> /dev/null || (cd "${gld_dir}" ; git pull)

# Build pyvvo.
echo "Building pyvvo container...\n"
docker build -t pyvvo:${tag} --build-arg MSCC=${mscc_path} --build-arg GLD=${gld_dir} .