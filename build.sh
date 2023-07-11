#!/bin/bash

# Uninstall the Python package using pip
yes | pip uninstall robingrad

# Read the version from pyproject.toml and store it in a variable
version=$(awk -F'"' '/version =/ {print $2}' pyproject.toml)

# Print the version
echo "Version: $version"

if [ -d "dist" ]; then
  rm -rf dist
fi

if [ -d "robingrad.egg-info" ]; then
  rm -rf robingrad.egg-info
fi

# build the package
python -m build

# install builded package 
cd dist
command="pip install robingrad-${version}.tar.gz"
eval $command


