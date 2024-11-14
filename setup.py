from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Mice Brain'
LONG_DESCRIPTION = ''

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="mice",
    version=VERSION,
    author="Luis Barba",
    author_email="<youremail@email.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    entry_points={

    }
)