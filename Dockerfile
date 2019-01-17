# NOTE: Do not directly call docker build. Instead run build.sh.

# Build on top of the base application container for GridAPPS-D, which
# is a Debian-based Python container and contains some GridAPPS-D
# utilities.
FROM gridappsd/app-container-base:latest

# Arguments for MySQL Connector/C (MSCC) and GridLAB-D (GLD) locations.
ARG MSCC
ARG GLD

# Work from pyvvo.
ENV PYVVO=/pyvvo

# TODO: In an ideal world, there would be a seperate docker container
# which contains GridLAB-D, rather than installing GridLAB-D into the
# pyvvo container. However, the level of effort to get that going simply
# isn't worth it right now.

# Setup other environment variables:
# MSCC --> MySQL Connector/C
# All libs are going into /pyvvo/lib except MSCC
ENV TEMP_DIR=/tmp/source \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PYVVO}/lib:/usr/local/mysql/lib \
    PATH=${PATH}:/${PYVVO}/bin \
    GLPATH=${PYVVO}/lib/gridlabd:${PYVVO}/share/gridlabd \
    CXXFLAGS=-I${PYVVO}/share/gridlabd \
    PACKAGES="autoconf automake g++ gcc libtool make"

# Copy MSCC into image.
COPY $MSCC /usr/local/mysql

# Copy GLD into image.
COPY $GLD ${TEMP_DIR}/gridlab-d

# Work in temporary directory.
WORKDIR ${TEMP_DIR}

# Install packages needed for software builds/installation
RUN perl -E "print '*' x 80" \
    && printf '\nInstalling packages for software builds/installation...\n' \
    && apt-get update && apt-get -y install --no-install-recommends ${PACKAGES} \
    && rm -rf /var/lib/opt/lists/* \
# Symlinks for MSCC. /usr/local/mysql is standard, GridLAB-D might need
# the mysql-connector-c as well?
    && perl -E "print '*' x 80" \
    && printf '\nCreating symlink for MySQL Connector/C...' \
    && ln -s /usr/local/mysql /usr/local/mysql-connector-c \
    && printf 'done.\n' \
# Install Xerces
    && perl -E "print '*' x 80" \
    && printf '\nInstalling Xerces...\n' \
    && cd ${TEMP_DIR}/gridlab-d/third_party \
    && tar -xzf xerces-c-3.2.0.tar.gz \
    && cd ${TEMP_DIR}/gridlab-d/third_party/xerces-c-3.2.0 \
    && ./configure --disable-static CFLAGS=-O2 CXXFLAGS=-O2 \
    && make \
    && make install \
# Install GridLAB-D
# TODO - should we run the GridLAB-D tests?
    && perl -E "print '*' x 80" \
    && printf '\nInstalling GridLAB-D...\n' \
    && cd ${TEMP_DIR}/gridlab-d \
    && autoreconf -isf \
    && ./configure --prefix=${PYVVO} --with-mysql=/usr/local/mysql --enable-silent-rules 'CFLAGS=-g -O2 -w' 'CXXFLAGS=-g -O2 -w -std=c++11' 'LDFLAGS=-g -O2 -w' \
    && make \
    && make install \
# Clean up source installs
    && perl -E "print '*' x 80" \
    && printf '\nCleaning up temporary directory...\n' \
    && cd "${PYVVO}" \
    && /bin/rm -rf "${TEMP_DIR}" \
# Remove software used for building.
    && perl -E "print '*' x 80" \
    && printf '\nRemoving packages...\n' \
    && apt-get purge -y --auto-remove ${PACKAGES} \
    && apt-get -y clean

WORKDIR $PYVVO

# Copy requirements.
COPY requirements.txt $PYVVO/requirements.txt

# Install requirements.
RUN pip install -r requirements.txt

# Copy application code.
COPY pyvvo /pyvvo/pyvvo

# Copy tests.
COPY tests /pyvvo/tests

# Work from code directory.
WORKDIR /pyvvo/pyvvo