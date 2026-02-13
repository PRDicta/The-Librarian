#!/bin/bash
# The Librarian — Dependency Installer
# Installs required Python packages for the Librarian memory system.

set -e

echo "Installing The Librarian dependencies..."

pip install --break-system-packages --quiet numpy rich anthropic 2>/dev/null || \
pip install --quiet numpy rich anthropic 2>/dev/null || \
pip install numpy rich anthropic

echo "Dependencies installed successfully."
