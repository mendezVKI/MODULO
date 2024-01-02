from os import path
from setuptools import setup, find_packages
import sys


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
if sys.version_info < min_version:
    error = """
modulo_vki does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

# with open(path.join(here, 'requirements.txt')) as requirements_file:
#     # Parse requirements.txt, ignoring any commented-out lines.
#     requirements = [line for line in requirements_file.read().splitlines()
#                     if not line.startswith('#')]


setup(
    name='modulo_vki',
    version='1.1.3',
    description="MODULO (MODal mULtiscale pOd) is a software developed at the von Karman Institute to perform Multiscale Modal Analysis of numerical and experimental data using the Multiscale Proper Orthogonal Decomposition (mPOD).",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="M. A. Mendez, D. Ninni, L. Schena",
    author_email='lorenzo.schena@vki.ac.be',
    url='https://github.com/mendezVKI/MODULO/tree/master/modulo_python_package/',
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    packages=find_packages(exclude=['docs', 'tests']),
    entry_points={
        'console_scripts': [
            # 'command = some.module:some_function',
        ],
    },
    include_package_data=True,
    package_data={
        'modulo': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=[
            "tqdm==4.59.0",
            "sklearn==0.0",
            "numpy==1.22.0",
            "scipy==1.5.4",
            "scikit-learn==0.24.1",
            "ipykernel==5.5.3",
            "ipython==8.10.0",
            "ipython-genutils==0.2.0",
            "ipywidgets==7.6.3",
            "matplotlib==3.3.4",
            "numpydoc==1.1.0",
    ],
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
