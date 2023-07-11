#!/bin/bash

# Uninstall the Python package using pip
yes | pip uninstall robin

# Read the version from pyproject.toml and store it in a variable
version=$(awk -F'"' '/version =/ {print $2}' pyproject.toml)

# Print the version
echo "Version: $version"

if [ -d "dist" ]; then
  rm -rf dist
fi

if [ -d "robin.egg-info" ]; then
  rm -rf robin.egg-info
fi

# build the package
python -m build

# install builded package 
cd dist
command="pip install robin-${version}.tar.gz"
eval $command


