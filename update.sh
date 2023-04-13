#!/bin/bash
if [ -d .git ]; then
    git fetch --prune
    git reset --hard origin/main
else
    git init
    git remote add origin
    git fetch --prune
    git reset --hard origin/main