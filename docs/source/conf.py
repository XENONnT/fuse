# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import fuse

project = 'XENON fuse'
copyright = '2024, fuse contributors, the XENON collaboration'

release = fuse.__version__
version = fuse.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'nbsphinx',
    ]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
#html_static_path = ['_static']

#Lets disable notebook execution for now
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'