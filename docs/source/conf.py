import os
import sys

# Allow Sphinx to find the project module
# sys.path.insert(0, os.path.abspath('../../'))

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------
project = 'MODULO'
author = 'Poletti, R., Schena, L., Ninni, D., Mendez, M.A.'
version = '2.1.0'
release = '2.1.0'
# e.g. "2024, von Karman Institute"
copyright = '2025 von Karman Institute'

# -- General configuration ---------------------------------------------------

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',       # Core: auto-documentation from docstrings
    'sphinx.ext.autosummary',   # Generate summary tables
    'sphinx.ext.intersphinx',   # Link to other projects' docs
    'sphinx.ext.napoleon',      # Google/NumPy style docstrings
    'sphinx.ext.mathjax',       # Render math via MathJax
    'sphinx.ext.viewcode',      # Link to source code
    'sphinx.ext.githubpages',   # Publish to GitHub Pages
    'sphinxcontrib.bibtex',     # Bibliographic citations
]

# Paths that contain templates, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# Files and patterns to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

# If true, show 'Created using Sphinx' in the footer
html_show_sphinx = False

# -- Bibliography configuration ---------------------------------------------
# Place your .bib files in this directory
bibtex_bibfiles = ['paper.bib']
bibtex_default_style = 'unsrt'

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    'python':   ('https://docs.python.org/3', None),
    'numpy':    ('https://numpy.org/doc/stable', None),
    'scipy':    ('https://docs.scipy.org/doc/scipy/reference', None),
    'pandas':   ('https://pandas.pydata.org/pandas-docs/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'sklearn':  ('https://scikit-learn.org/stable', None),
}

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = 'bysource'
autosummary_generate = True

def setup(app):
    # Example: add custom CSS/JS
    app.add_css_file('custom.css')

