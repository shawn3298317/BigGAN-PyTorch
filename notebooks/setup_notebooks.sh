#!/bin/bash

# Start from directory of script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Set up symlinks for the example notebooks
ln -sfn ../logs .
ln -sfn ../samples .
ln -sfn ../weights .
