import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../source'))

#import sphinx_rtd_theme

#html_theme = "sphinx_rtd_theme"
# -- Project information -----------------------------------------------------

project = 'Vision'
copyright = '2023, Inspecity Space Labs'
author = 'R Narwar'

# The full version, including alpha/beta/rc tags
release = '1.3'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#extensions = ["sphinx_rtd_theme",
#              "sphinx.ext.autodoc",
#              "sphinx.ext.coverage",
#              "sphinx.ext.napoleon"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static','_figures']

html_theme_options = {
    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#222A35',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Get the direct path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Add the HTML logo that displays on the top-left panel.
#html_logo = dir_path + '/_static/leogps_favicon2.png'

# Add the HTML logo.
#html_favicon = '_static/leogps_favicon.png'
