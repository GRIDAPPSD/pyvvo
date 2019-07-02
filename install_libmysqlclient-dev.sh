#!/usr/bin/env bash
# Helper script used by build.sh to install libmysqlclient-dev.sh
# This should be run after apt-get update has already been called.
# INPUTS:
#   1) Name of the "mysql-atp-config..." .deb file.
#   2) Dependencies for running dpkg -i <deb file>

# Collect input.
deb_file=${1:-mysql-apt-config_0.8.13-1_all.deb}
deps=${2:-"lsb-release wget gnupg"}

perl -E "print '*' x 80"
printf "\nStarting the process of installing libmysqlclient-dev...\n"
printf "Starting by installing dependencies.\n"
apt-get -y --no-install-recommends install ${deps}

printf "Setting up debconf selections so there are no prompts...\n"

# You can find the selections necessary by installing debconf-utils
# and using "debconf-get-selections | grep mysql" after calling
# "dpkg -i <package>" and walking through the prompts.
export DEBIAN_FRONTEND="noninteractive"
echo "mysql-apt-config mysql-apt-config/select-server select mysql-8.0" | debconf-set-selections
echo "mysql-apt-config mysql-apt-config/select-tools select Enabled" | debconf-set-selections
echo "mysql-apt-config mysql-apt-config/select-product select Ok" | debconf-set-selections

printf "Configuring the MySQL repository.\n"
dpkg -i ${deb_file}

printf "Installing libmysqlclient-dev...\n"
apt-get update
apt-get -y --no-install-recommends install libmysqlclient-dev
printf "Done installing libmysqlclient-dev. Cleaning up...\n"
apt-get purge -y --autoremove ${deps}
printf "libemysqlclient installation dependencies removed.\n"
