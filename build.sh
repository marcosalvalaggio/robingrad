#!/bin/bash

#!/bin/bash

# Read the version from pyproject.toml and store it in a variable
version=$(awk -F'"' '/version =/ {print $2}' pyproject.toml)

# Print the version
echo "Version: $version"
