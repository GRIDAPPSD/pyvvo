# Use slim, which is built on a slim Debian image.
FROM python:slim

# Work from pyvvo.
ENV PYVVO=/pyvvo

# Setup other environment variables:
# MSCC --> MySQL Connector/C
# All libs are going into /pyvvo/lib except MSCC
ENV MSCC_VERSION=6.1.11 \
    TEMP_DIR=/tmp/source \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PYVVO}/lib:/usr/local/mysql/lib \
    PATH=${PATH}:/${PYVVO}/bin \
    GLPATH=${PYVVO}/lib/gridlabd:${PYVVO}/share/gridlabd \
    CXXFLAGS=-I${PYVVO}/share/gridlabd \
    PACKAGES="autoconf automake g++ gcc git libtool make wget"

# Define the MSCC archive name.
ENV MSCC_DIR=mysql-connector-c-${MSCC_VERSION}-linux-glibc2.12-x86_64
ENV MSCC_ARCHIVE=${MSCC_DIR}.tar.gz

# Define full MSCC download URL.
ENV MSCC_DOWNLOAD=https://dev.mysql.com/get/Downloads/Connector-C/${MSCC_ARCHIVE}

# Work in temporary directory.
WORKDIR ${TEMP_DIR}

# Install packages needed for software builds/installation
RUN perl -E "print '*' x 80" \
    && printf '\nInstalling packages for software builds/installation...\n' \
    && apt-get update && apt-get -y install ${PACKAGES} \
    && rm -rf /var/lib/opt/lists/* \
# Install MySQL Connector/C.
    && perl -E "print '*' x 80" \
    && printf '\nInstalling MySQL Connector/C...\n' \
    && wget ${MSCC_DOWNLOAD} \
    && tar -C /usr/local -zxf  ${MSCC_ARCHIVE} \
# Symlinks for MSCC. /usr/local/mysql is standard, GridLAB-D might need
# the mysql-connector-c as well?
    && ln -s /usr/local/${MSCC_DIR} /usr/local/mysql \
    && ln -s /usr/local/${MSCC_DIR} /usr/local/mysql-connector-c \
# Install Xerces
    && cd $TEMP_DIR \
    && git clone https://github.com/gridlab-d/gridlab-d.git -b develop --single-branch \
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

# Work from code directory
WORKDIR /pyvvo/pyvvo