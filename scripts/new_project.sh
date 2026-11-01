#!/bin/bash
# Usage: ./scripts/new_project.sh <project-name>
# Creates a new project from the default template

if [ -z "$1" ]; then
    echo "Usage: $0 <project-name>"
    exit 1
fi

TEMPLATE_DIR="$(dirname "$0")/.."
PROJECTS_DIR="$(dirname "$TEMPLATE_DIR")/../projects"
TARGET="$PROJECTS_DIR/$1"

if [ -d "$TARGET" ]; then
    echo "Error: Project '$1' already exists."
    exit 1
fi

cp -r "$TEMPLATE_DIR" "$TARGET"
cd "$TARGET" && git init
echo "Project '$1' created at $TARGET"
