# NOTE: Do not directly call docker build. Instead run build.sh.

# Use multistage builds!
FROM debian:stretch AS builder

# Setup environment variables for building.
ENV PYVVO=/pyvvo \
    TEMP_DIR=/tmp/pyvvo

WORKDIR ${TEMP_DIR}

# Arguments for MySQL Connector/C (MSCC) and GridLAB-D (GLD) locations.
ARG MSCC
ARG GLD

# Add GridLAB-D and MSCC archives. Hard-code build directory.
ADD $GLD $MSCC ${TEMP_DIR}/

ENV CXXFLAGS=-I${PYVVO}/share/gridlabd \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PYVVO}/lib:/usr/local/mysql/lib \
    PATH=${PATH}:/${PYVVO}/bin \
    BUILD_PACKAGES="autoconf automake g++ libtool make gcc"

# TODO: In an ideal world, there would be a seperate docker container
# which contains GridLAB-D, rather than installing GridLAB-D into the
# pyvvo container. However, the level of effort to get that going simply
# isn't worth it right now.

# Move MSCC to /usr/local/mysql.
ARG MSCC_DIR_NAME
# Install necessary packages.
RUN perl -E "print '*' x 80" \
    && printf '\nInstalling packages for software builds/installation...\n' \
    && apt-get update && apt-get -y install --no-install-recommends ${BUILD_PACKAGES} \
    && rm -rf /var/lib/opt/lists/* \
    # Move the MSCC to the proper directory.
    && mv ${TEMP_DIR}/${MSCC_DIR_NAME}/ /usr/local/mysql \
    # Symlinks for MSCC. /usr/local/mysql is standard, GridLAB-D might need
    # the mysql-connector-c as well?
    && perl -E "print '*' x 80" \
    && printf '\nCreating symlink for MySQL Connector/C...' \
    && ln -s /usr/local/mysql /usr/local/mysql-connector-c \
    && printf 'done.\n' \
    # Make PyVVO directory.
    && mkdir ${PYVVO} \
    # Install Xerces
    && perl -E "print '*' x 80" \
    && printf '\nInstalling Xerces...\n' \
    && cd ${TEMP_DIR}/gridlab-d/third_party \
    && tar -xzf xerces-c-3.2.0.tar.gz \
    && cd ${TEMP_DIR}/gridlab-d/third_party/xerces-c-3.2.0 \
    && ./configure --prefix=${PYVVO} --disable-static CFLAGS=-O2 CXXFLAGS=-O2 \
    && make -j $(($(nproc) + 1)) \
    && make -j $(($(nproc) + 1)) install \
    # Install GridLAB-D
    # TODO - should we run the GridLAB-D tests?
    && perl -E "print '*' x 80" \
    && printf '\nInstalling GridLAB-D...\n' \
    && cd ${TEMP_DIR}/gridlab-d \
    && autoreconf -isf \
    && ./configure --prefix=${PYVVO} --with-mysql=/usr/local/mysql --with-xerces=${PYVVO} --enable-silent-rules 'CFLAGS=-g -O2 -w' 'CXXFLAGS=-g -O2 -w -std=c++11' 'LDFLAGS=-g -O2 -w' \
    && make -j $(($(nproc) + 1)) \
    && make -j $(($(nproc) + 1)) install \
    # Clean up source installs
    && perl -E "print '*' x 80" \
    && printf '\nCleaning up temporary directory...\n' \
    && cd "${PYVVO}" \
    && /bin/rm -rf "${TEMP_DIR}" \
    # Remove software used for building.
    && perl -E "print '*' x 80" \
    && printf '\nRemoving packages...\n' \
    && apt-get purge -y --auto-remove ${BUILD_PACKAGES} \
    && apt-get -y clean

# Build on top of the base application container for GridAPPS-D, which
# is a Debian-based Python container and contains some GridAPPS-D
# utilities.
# TODO: update to use latest or develop.
FROM gridappsd/app-container-base:blthayer

# Work from pyvvo.
ENV PYVVO=/pyvvo
WORKDIR ${PYVVO}

# Setup other environment variables:
# MSCC --> MySQL Connector/C
# All libs are going into /pyvvo/lib except MSCC
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYVVO}/lib:/usr/local/mysql/lib \
    PATH=${PATH}:/${PYVVO}/bin \
    GLPATH=${PYVVO}/lib/gridlabd:${PYVVO}/share/gridlabd

# Copy in stuff from our builder.
COPY --from=builder ${PYVVO} ${PYVVO}
COPY --from=builder /usr/local/mysql /usr/local/mysql-connector-c /usr/local/

# Copy requirements.
COPY requirements.txt ${PYVVO}/requirements.txt

# Install requirements.
RUN pip install -r requirements.txt

# Add the pyvvo application files.
ARG PYVVO_ARCHIVE
ADD ${PYVVO_ARCHIVE} ${PYVVO}/pyvvo

# Work from code directory.
WORKDIR ${PYVVO}/pyvvo
