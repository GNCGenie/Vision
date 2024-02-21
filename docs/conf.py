import os
import sys

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../source'))

import sphinx_rtd_theme

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
html_static_path = ['_static', '_figures']  # Remove unnecessary commas

html_theme_options = {
    'analytics_id': 'G-XXXXXXXXXX',  # Provided by Google in your dashboard
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Logo and favicon paths (update correctly)
# html_logo = dir_path + '/_static/leogps_favicon2.png'
# html_favicon = '_static/leogps_favicon.png'
