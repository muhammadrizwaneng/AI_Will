#!/bin/bash

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Installing Python..."
    sudo apt update
    sudo apt install -y python3
else
    echo "Python is already installed."
fi

# Install pip
if ! command -v pip &> /dev/null; then
    echo "Pip is not installed. Installing Pip..."
    sudo apt update
    sudo apt install -y python3-pip
else
    echo "Pip is already installed."
fi

# Install required libraries
echo "Installing required libraries..."
sudo -H pip3 install flask transformers pandas torch

echo "Setup complete."
