#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import subprocess

def run_and_update(executable, ref_file, update_refs):
    p = subprocess.Popen([ executable ], stdout=subprocess.PIPE)
    p.wait()
    stdout, _ = p.communicate()
    
    reference = open(ref_file, 'rb').read()

    if stdout != reference:
        # Update the reference if applicable
        if update_refs:
            open(ref_file, 'wb').write(stdout)
            return
        
        sys.exit(1)

# Run the first argument and compare it to the file provided in the second argument
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} EXECUTABLE REFERENCE_FILE")
        print("Run with env variable UPDATE_REFERENCES set to update the reference files.")
        sys.exit(1)

    executable = sys.argv[1]
    ref_file = sys.argv[2]
    update_refs = "UPDATE_REFERENCES" in os.environ

    run_and_update(executable, ref_file, update_refs)