"""Helper script to build the PyVVO documentation. This script does
some prep work before simply calling "make html." Be sure to add LaTex
files which do not need to be individually compiled to the EXCLUDE list.
"""
import os
import subprocess
import re
import argparse

# List .tex files in the latex directory that should not be compiled
# themselves. Don't include file extensions.
EXCLUDE = ['flow_base']

# Directories of latex and rst files.
LATEX_DIR = 'latex'
RST_DIR = 'rst_latex'


def stars(n=80):
    """Simply print some stars."""
    print('*' * n)


def aux2dict(f_in):
    """Read a LaTex .aux file to get the labels.

    :param f_in: Name of aux file without directory or extension.
        Assumed to be in the ``latex`` directory.
    """
    with open(os.path.join(LATEX_DIR, f_in + '.aux'), 'r') as f:
        # Read.
        file_str = f.read()

    # Extract the label name and the label value.
    results = re.findall(r'(?:\{)(.*)(?:\})(?:\{\{)(.*)(?:\}\{.*\}\})',
                         file_str)

    # Map it to a dictionary.
    return {x[0]: x[1] for x in results}


def update_rst(f_in):
    """Read a .rst file and update the references it makes according to
    the ref_dict. It is assumed the .rst file is in this directory. It
    is also assumed that the .rst file follows this convention: All
    references will be denoted with a comment like (but with only a
    single slash, the second slash you see in the source is just so that
    the rst parser for this script isn't upset):

    ``.. \\ref{flow:start}``

    On the next line following the reference line will be the rendering
    of the reference (which is what will be updated), like so:

    ``(a)``

    :param f_in: .rst file, but without the extension. E.g. "main_loop"
    """
    # Get a dictionary of references.
    ref_dict = aux2dict(f_in)

    # Start by reading the .rst file.
    full_path = os.path.join(RST_DIR, f_in + '.rst')
    with open(full_path, 'r') as f:
        rst = f.read()

    # Now loop over the dictionary and make updates.
    for ref, val in ref_dict.items():
        #
        rst = re.sub(r'\\ref\{{{}\}}'.format(ref),
                     r'({})'.format(val), rst)

    # Replace.
    with open(full_path, 'w') as f:
        f.write(rst)

    # Done.
    return None


def main(checkout):
    # Get listing of .tex files without the extension.
    tex_files = [os.path.splitext(x)[0] for x in os.listdir(LATEX_DIR)
                 if x.endswith('.tex')]

    # Compile each .tex file, then update the references in the
    # corresponding .rst file.
    for tf in tex_files:
        # Skip files in the EXCLUDE list.
        if tf in EXCLUDE:
            continue

        # Compile and get svg files.
        stars()
        print('Running tex2svg for {}'.format(tf))
        subprocess.run(['./tex2svg.sh', tf], cwd=LATEX_DIR)
        stars()

        # Update references in .rst files.
        update_rst(tf)

    # Finally, build the documentation.
    stars()
    print('Building the documentation.')
    subprocess.run(['make', 'html'])
    stars()

    # (Maybe) check out the files in rst_latex.
    if checkout:
        stars()
        print('Checking out files in {}.'.format(RST_DIR))
        subprocess.run(['git', 'checkout', './*'], cwd=RST_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkout', action='store_true',
        help='Set this flag if you wish to run the "git checkout ./*" '
             'operation inside the "rst_latex" directory at the end of '
             'the script. This will help avoid the annoyances related '
             'to this script constantly changing the .rst files there.')

    args = parser.parse_args()

    main(checkout=args.checkout)
