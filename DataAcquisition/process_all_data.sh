#!/bin/bash

# Find all directories containing 'process_data.py' and run them in parallel
find . -name 'process_data.py' -print0 | xargs -0 -I {} -P 0 sh -c 'cd $(dirname {}); python3 process_data.py'