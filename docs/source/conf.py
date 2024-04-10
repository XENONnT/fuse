# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import fuse

project = "XENON fuse"
copyright = "2024, fuse contributors, the XENON collaboration"

release = fuse.__version__
version = fuse.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []  # type: ignore


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
# html_static_path = ['_static']

# Lets disable notebook execution for now
nbsphinx_allow_errors = True
nbsphinx_execute = "never"


def setup(app):
    # app.add_css_file('css/custom.css')
    # Hack to import something from this dir. Apparently we're in a weird
    # situation where you get a __name__  is not in globals KeyError
    # if you just try to do a relative import...
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from build_release_notes import convert_release_notes
    from build_plugin_pages import build_all_pages

    convert_release_notes()
    build_all_pages()
