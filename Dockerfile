# Use stretch, which is build on the latest Debian.
FROM python:stretch

# Work from pyvvo.
ENV PYVVO=/pyvvo
WORKDIR $PYVVO

# Copy requirements.
COPY requirements.txt /pyvvo/requirements.txt

# Install requirements.
RUN pip install -r requirements.txt

# Copy application code.
COPY pyvvo /pyvvo/pyvvo

# Work from code directory
WORKDIR /pyvvo/pyvvo

# Run Python as an entrypoint.
ENTRYPOINT python
CMD -m parse_glm
