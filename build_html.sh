#!/bin/bash

set -e

rm -rf _build/rst _build/html
d2lbook build rst
cp static/frontpage.html _build/rst/

d2lbook build html
# mkdir _build/html/_images/
cp -r static/image/* _build/html/_images/

cp -r static/template/material.blue-deep_orange.min.css _build/html/_static/material-design-lite-1.3.0/
cp -r static/template/sphinx_materialdesign_theme.css _build/html/_static/