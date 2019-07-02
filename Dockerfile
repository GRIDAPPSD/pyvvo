# NOTE: Do not directly call docker build. Instead run build.sh.
#
# Build on top of the base application container for GridAPPS-D, which
# is a Debian-based Python container and contains some GridAPPS-D
# utilities.

# Work around docker bug:
# https://stackoverflow.com/questions/51981904/cant-build-docker-multi-stage-image-using-arg-in-copy-instruction
ARG TAG
FROM pyvvo-base:${TAG:-latest} as base

FROM gridappsd/app-container-base:develop

# Work from pyvvo.
ENV PYVVO=/pyvvo
WORKDIR ${PYVVO}

# Setup other environment variables:
# MSCC --> MySQL Connector/C
# All libs are going into /pyvvo/lib except MSCC
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYVVO}/lib:/usr/local/mysql/lib \
    PATH=${PATH}:/${PYVVO}/bin \
    GLPATH=${PYVVO}/lib/gridlabd:${PYVVO}/share/gridlabd

# Copy in stuff from our base container.
ARG TAG
COPY --from=base ${PYVVO} ${PYVVO}

# Arguments for mysql-apt-config.
ARG mysql_apt
ENV mysql_apt=${mysql_apt}
ARG mysql_apt_deps
ENV mysql_apt_deps=${mysql_apt_deps}

# Copy files in.
COPY requirements.txt setup.py install_libmysqlclient-dev.sh ${mysql_apt} ${PYVVO}/

# Install libmysqlclient-dev and Python requirements.
RUN BUILD_DEPS="build-essential libssl-dev python3-dev" \
    && apt-get update \
    && ./install_libmysqlclient-dev.sh ${mysql_apt} "${mysql_apt_deps}" \
    && rm ${mysql_apt} \
    && rm install_libmysqlclient-dev.sh \
    && apt-get -y --no-install-recommends install ${BUILD_DEPS} \
    && pip install --no-cache-dir -r requirements.txt \
    # Apt cleaning.
    && rm -rf /var/lib/opt/lists/* \
    && apt-get purge -y --auto-remove ${BUILD_DEPS} \
    && apt-get -y clean

# Add the pyvvo application files.
ARG PYVVO_ARCHIVE
ADD ${PYVVO_ARCHIVE} ${PYVVO}/pyvvo

# Work from code directory.
WORKDIR ${PYVVO}/pyvvo
