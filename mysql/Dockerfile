FROM mysql:latest

# The MySQL image errors out without secure-file-priv set up. Make a
# directory for it. Note this corresponds to the secure-file-priv
# setting in my.cnf.
RUN mkdir -p /var/lib/mysql-files \
    && chown mysql:mysql /var/lib/mysql-files \
    && chmod 740 /var/lib/mysql-files

# Instead of doing environment setup here in the Dockerfile, we'll
# handle this at the docker-compose level so that the pyvvo container
# can easily access everything without hard-coding.
#
## Setup environment. To make this more secure in the future we'll want
## to use Docker secrets. For now, this is just fine.
#ENV MYSQL_RANDOM_ROOT_PASSWORD='yes' \
#    MYSQL_DATABASE='pyvvo' \
#    MYSQL_USER='pyvvo' \
#    MYSQL_PASSWORD=''

# Copy in configuration file.
COPY my.cnf /etc/mysql/my.cnf