# Use the latest slim python.
FROM python:slim

# Work from pyvvo.
WORKDIR /pyvvo

# Copy requirements.
COPY requirements.txt /pyvvo/

# Install requirements.
RUN pip install -r requirements.txt

# Copy application code.
COPY pyvvo/ /pyvvo/pyvvo

# Run Python as an entrypoint.
ENTRYPOINT python