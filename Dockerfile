# Use the latest slim python.
FROM python:slim

# Work from pyvvo.
WORKDIR /pyvvo

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
