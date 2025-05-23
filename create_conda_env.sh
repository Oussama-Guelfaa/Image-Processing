#!/bin/bash

# Script to create and initialize a Conda environment for a project
# Usage: ./create_conda_env.sh [env_name] [python_version]

# Default values
ENV_NAME=${1:-"project-env"}
PYTHON_VERSION=${2:-"3.12"}

# Create the Conda environment
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate the environment
echo "Activating environment..."
source /opt/anaconda3/bin/activate $ENV_NAME

# Install common packages
echo "Installing common packages..."
conda install -y numpy matplotlib scikit-image scikit-learn opencv

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Create VS Code settings file for this project
echo "Creating VS Code settings for this project..."
cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "/opt/anaconda3/envs/$ENV_NAME/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.terminal.activateEnvInCurrentTerminal": true,
    "terminal.integrated.env.osx": {
        "PATH": "/opt/anaconda3/envs/$ENV_NAME/bin:\${env:PATH}"
    }
}
EOF

echo "Done! Environment '$ENV_NAME' is ready to use."
echo "To activate this environment manually, run: source /opt/anaconda3/bin/activate $ENV_NAME"
