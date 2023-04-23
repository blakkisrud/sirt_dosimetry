# -*- coding: utf-8 -*-
"""
Script to perform conversion from a count map to absorbed dose map
Usage:

python create_dose_map --input_path input_path --output_path output_path shunt_factor shunt
Input variables:
input_path: Path to the count map - have to be an nrrd-file
output_path: Path to the output dose map file
shunt_factor: The shunt factor given in an interval of 0.0 and 1.0. If
omitted a default value of 0.0 will be used
@author: johbla@ous-hf.no/blakkisrud@gmail.com
"""

# =============================================================================
# Import statements
# =============================================================================

import argparse
import nrrd
import numpy as np
import sirt_functions as sf

if __name__ == "__main__":
    # =============================================================================
    # Argument parsing
    # =============================================================================

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Converting count map to dose map")

    # Adding arguments
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--shunt_factor', type=float, required=False)

    # Parse
    args = parser.parse_args()

    # If shunt factor is not given, a shunt of 0.0 is assumed
    if args.shunt_factor == None:
        shunt_factor = 0.0
    else:
        shunt_factor = args.shunt_factor

        
    print("Creating dosemap...")
    sf.dose_map_func(args.input_path, output_path=args.output_path, shunt_factor=shunt_factor)
