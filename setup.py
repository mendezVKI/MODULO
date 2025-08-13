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

with open(path.join(here, 'README_pypi.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()
    

setup(
    name='modulo_vki',
    version='2.1.4',
    description="MODULO (MODal mULtiscale pOd) is a software developed at the von Karman Institute to perform "
                "Multiscale Modal Analysis of numerical and experimental data.",
    long_description=readme,
    long_description_content_type='text/x-rst',
    author=["R. Poletti","L. Schena", "D. Ninni", "M. A. Mendez"],
    author_email='mendez@vki.ac.be',
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
        "tqdm",
        "numpy",
        "scipy",
        "scikit-learn",
        "ipykernel",
        "ipython",
        "ipython-genutils",
        "ipywidgets",
        "matplotlib",
    ],
    extra_requires={
        'tutorials': [
            "pyvista",
            "imageio",
        ]
    },
    license='BSD (3-clause)',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)