# NOTE: Do not directly call docker build. Instead run build.sh.
#
# Build on top of the base application container for GridAPPS-D, which
# is a Debian-based Python container and contains some GridAPPS-D
# utilities.

# Work around docker bug:
# https://stackoverflow.com/questions/51981904/cant-build-docker-multi-stage-image-using-arg-in-copy-instruction
ARG TAG
FROM pyvvo-base:${TAG:-latest} as base

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

# Copy in stuff from our base container..
ARG TAG
COPY --from=base ${PYVVO} ${PYVVO}
ARG TAG
COPY --from=base /usr/local/mysql /usr/local/mysql-connector-c /usr/local/

# Copy requirements.
COPY requirements.txt ${PYVVO}/requirements.txt

# Install requirements.
RUN pip install --no-cache-dir -r requirements.txt

# Add the pyvvo application files.
ARG PYVVO_ARCHIVE
ADD ${PYVVO_ARCHIVE} ${PYVVO}/pyvvo

# Work from code directory.
WORKDIR ${PYVVO}/pyvvo
