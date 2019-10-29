from setuptools import setup, find_packages

__version__ = '0.0.1'

packages = find_packages()

# I replaced the requirements in requirements.txt with a simple '.'
# https://stackoverflow.com/questions/14399534/reference-requirements-txt-for-the-install-requires-kwarg-in-setuptools-setup-py
install_requires = ['coverage', 'mysqlclient', 'numpy', 'pandas',
                    'python-dateutil', 'scipy', 'simplejson', 'scikit-learn',
                    'stomp.py']

setup(
    name='pyvvo',
    version=__version__,
    author='Brandon Thayer',
    author_email='brandon.thayer@pnnl.gov',
    description='Volt/var optimization application for GridAPPS-D' + u'\u2122',
    long_description='# TODO',
    long_description_content_type='text/markdown',
    url='https://github.com/GRIDAPPSD/pyvvo',
    packages=packages,
    install_requires=install_requires,
    test_suite='tests',
    license='BSD 2-Clause License',
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.7"
    ],
    include_package_data=True,
    package_data={
        'pyvvo': ['log_config.json']
    }
)
