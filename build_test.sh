#!/bin/bash

# Define the package name, version, and Python version
PACKAGE_NAME="edterm"
VERSION="0.1.0"
PYTHON_VERSION="3.11"

# Define the test environment name
ENV_NAME="edterm-test_env"

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to find conda installation
find_conda() {
    # Use `which conda` to find the Conda executable
    local conda_path=$(which conda)
    if [[ -z $conda_path ]]; then
        echo -e "\n----- ${RED}No conda installation found.${NC} -----" >&2
        exit 1
    fi
    echo $(dirname $(dirname $conda_path))
}

# Use the function to find conda
CONDA_PATH=$(find_conda)

# Source the conda initialization script
. $CONDA_PATH/etc/profile.d/conda.sh


# Define a cleanup function
cleanup() {
    echo -e "\n----- ${GREEN}Cleaning up...${NC} -----"
    conda build purge
    conda activate base
    conda env remove -n $ENV_NAME -y
    rm -rf dist/
    rm -rf build/
    rm -rf *.egg-info
    echo -e "\n----- ${GREEN}Environment cleanup complete.${NC} -----"
}

# Define a help function
help() {
    echo "Usage: $0 [--clean | --no-cleanup | --help]"
    echo
    echo "Options:"
    echo "  --clean       Only run the cleanup function and then exit."
    echo "  --no-cleanup  Skip the cleanup even if the build fails."
    echo "  --help        Show this help message and exit."
    echo
    echo "If no options are provided, the script will clean up the environment at the end, whether the build succeeds or fails."
}

# Check for flags
case "$1" in
    --clean)
        cleanup
        exit 0
        ;;
    --no-cleanup)
        # If --no-cleanup flag is set, do not run the cleanup function
        ;;
    --help)
        help
        exit 0
        ;;
    *)
        if [ -n "$1" ]; then
            echo -e "\n----- ${RED}Unrecognized option: $1${NC} -----"
            echo
            help
            exit 1
        fi
        ;;
esac

# If any command fails and --no-cleanup flag is not set, run the cleanup function
if [ "$1" != "--no-cleanup" ]; then
    trap cleanup EXIT
fi

echo -e "\n----- ${GREEN}Creating a new conda environment...${NC} -----"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo -e "\n----- ${GREEN}Activating the conda environment...${NC} -----"
source activate $ENV_NAME

echo -e "\n----- ${GREEN}Building and installing $PACKAGE_NAME for pip..${NC} -----"


# Clean previous builds
rm -rf dist/
rm -rf build/
rm -rf *.egg-info

# Build the package
python setup.py sdist bdist_wheel

# Install the package locally for development
pip install -e .

echo -e "\n----- ${GREEN}Running pip package tests...${NC} -----"
# Here you can add commands to run your tests, e.g., pytest

echo -e "\n----- ${GREEN}Building and installing $PACKAGE_NAME for Conda...${NC} -----"

# Build the Conda package
conda build .

# Find the package location
CONDA_PACKAGE=$(conda build . --output)

# Install the package into the current environment
conda install --use-local $CONDA_PACKAGE -c conda-forge

echo -e "\n----- ${GREEN}Running Conda package tests...${NC} -----"
# Similar to above, add commands to run your tests

echo -e "\n----- ${GREEN}Build and installation complete.${NC} -----"
