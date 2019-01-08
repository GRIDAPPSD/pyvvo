#!/usr/bin/env bash
# Use this to build pyvvo.
#
# Prereqs: git, wget
#
# Arguments:
#   - tag: tag for image. Image name will be pyvvo:<tag>

# Collect arguments.
tag=$1

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
    wget -P "${build_dir}" --no-clobber "https://dev.mysql.com/get/Downloads/Connector-C/${mscc_archive}"
    tar -zxf "${build_dir}/${mscc_archive}" --directory "${build_dir}"
fi

# Clone GridLAB-D, or pull the latest.
git clone https://github.com/gridlab-d/gridlab-d.git -b develop --single-branch "${gld_dir}" 2> /dev/null || (cd "${gld_dir}" ; git pull)

# Build pyvvo.
docker build -t pyvvo:${tag} --build-arg MSCC=${mscc_path} --build-arg GLD=${gld_dir} .