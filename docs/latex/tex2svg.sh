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

# Build via latex. Run twice to ensure references work out.
latex ${file}.tex
latex ${file}.tex

# Convert to svg.
dvisvgm --font-format=woff2 ${file}.dvi
