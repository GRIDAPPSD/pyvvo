#!/bin/bash
# Helper script to compile a .tex file to .dvi then convert to .svg.
# ARGUMENTS:
#   filename: Name of .tex file, sans the extension (e.g. main_loop)
#
# PREREQUISITES:
#   latex
#   dvisvgm

die () {
    echo >&2 "$@"
    exit 1
}

[ "$#" -eq 1 ] || die "1 argument required, $# provided"

# Collect arguments.
file=${1}

# Ensure script fails on error.
set -e

# Build via latex. Run twice to ensure references work out. Run the
# first time in draftmode to reduce I/O.
latex -halt-on-error -quiet -interaction=nonstopmode ${file}.tex
latex -halt-on-error -quiet -interaction=nonstopmode ${file}.tex

# Convert to svg.
dvisvgm --font-format=woff2 --verbosity=3 ${file}.dvi
