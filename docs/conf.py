import os
import sys
import subprocess

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath('..'))

# Install sphinx_rtd_theme if not already present
subprocess.call(['pip', 'install', 'sphinx-rtd-theme'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Project information
project = 'Vision'
copyright = '2023, Inspecity Space Labs'
author = 'R Narwar'

# The full version, including alpha/beta/rc tags
release = 'v0.1.0'

# General configuration
extensions = [
    "sphinx_rtd_theme",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']  # Add your static files here
html_theme_options = {
    'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
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

# Additional configurations (optional)
# For autodoc generation
# source_suffix = '.rst'  # If using RST files
# master_doc = 'index'  # Main RST file

# For autoapi generation
# add_module_paths = ['path/to/your/source']

# For other extensions, refer to their documentation

## Build the documentation
#if __name__ == '__main__':
#    from sphinx.cmdline import build
#    build.main(["conf.py"])
