#!/bin/bash
# The Librarian — Dependency Installer
# Checks environment and installs required Python packages.

set -e

REQUIRED_PACKAGES=("numpy" "rich" "anthropic")
MISSING=()

# Check Python availability
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo '{"status": "error", "message": "Python is not installed. The Librarian requires Python 3.9+. Please install Python from https://python.org and try again."}'
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)
PYTHON_VERSION=$($PYTHON --version 2>&1)

# Check each package
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! $PYTHON -c "import $pkg" 2>/dev/null; then
        MISSING+=("$pkg")
    fi
done

# If nothing is missing, we're done
if [ ${#MISSING[@]} -eq 0 ]; then
    echo '{"status": "ok", "message": "All dependencies already installed.", "python": "'"$PYTHON_VERSION"'"}'
    exit 0
fi

echo "Installing: ${MISSING[*]}..."

# Try install with progressively fewer flags
if pip install --break-system-packages --quiet "${MISSING[@]}" 2>/dev/null; then
    echo '{"status": "ok", "message": "Dependencies installed successfully.", "installed": "'"${MISSING[*]}"'", "python": "'"$PYTHON_VERSION"'"}'
elif pip install --quiet "${MISSING[@]}" 2>/dev/null; then
    echo '{"status": "ok", "message": "Dependencies installed successfully.", "installed": "'"${MISSING[*]}"'", "python": "'"$PYTHON_VERSION"'"}'
elif pip3 install --break-system-packages --quiet "${MISSING[@]}" 2>/dev/null; then
    echo '{"status": "ok", "message": "Dependencies installed successfully.", "installed": "'"${MISSING[*]}"'", "python": "'"$PYTHON_VERSION"'"}'
elif pip3 install --quiet "${MISSING[@]}" 2>/dev/null; then
    echo '{"status": "ok", "message": "Dependencies installed successfully.", "installed": "'"${MISSING[*]}"'", "python": "'"$PYTHON_VERSION"'"}'
else
    echo '{"status": "error", "message": "Could not install packages: '"${MISSING[*]}"'. Try running manually: pip install '"${MISSING[*]}"'"}'
    exit 1
fi
