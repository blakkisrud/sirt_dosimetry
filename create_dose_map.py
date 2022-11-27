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


# Constants
TISSUE_DENSITY = 1.05  # grams per cubic cm
DOSE_CONSTANT = 50  # Given admin activity in GBq and mass in Kg


def dose_map_func(input_path, output_path="", shunt_factor=0.0):

    image_array, image_header = nrrd.read(input_path, index_order='C')  # SPECT

    space_dirs = (image_header["space directions"])

    x_dim = np.abs((space_dirs[0, 0])) / 10
    y_dim = np.abs((space_dirs[1, 1])) / 10
    z_dim = np.abs((space_dirs[2, 2])) / 10

    voxel_mass = (x_dim * y_dim * z_dim * TISSUE_DENSITY) / 1e3  # Voxel mass in kg

    image_as_double = image_array.astype(np.float32)

    sum_of_counts = np.sum(image_as_double)

    # Correct for shunt

    sum_of_counts = sum_of_counts / (1 - shunt_factor)

    fraction_image = image_as_double / sum_of_counts

    # Calculate and save dose map
    # Re-use the image header from the count-image

    dose_map = (fraction_image * DOSE_CONSTANT) / (voxel_mass);
    if output_path:
        nrrd.write(output_path, dose_map, header=image_header, index_order='C')
        print("Dose map saved as nrrd at", output_path)
    return dose_map

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
    dose_map_func(args.input_path, output_path=args.output_path, shunt_factor=shunt_factor)
