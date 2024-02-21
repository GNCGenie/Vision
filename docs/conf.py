import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../source'))

# Project information
project = 'Vision'
copyright = '2023, Inspecity Space Labs'
author = 'R Narwar'

# The full version, including alpha/beta/rc tags
release = '0.1'

# General configuration
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_figures']  # Remove unnecessary commas
