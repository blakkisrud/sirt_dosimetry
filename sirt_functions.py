"""

This file contains the functions used in the SIRT-dosimetry
calculations.

Author: johbla@ous-hf.no

"""

# =============================================================================
# Import statements
# =============================================================================

import argparse
import nrrd
import numpy as np
import yaml


# =============================================================================
# Functions
# =============================================================================

def read_constants(path_to_constants = "constants.yaml"):
    with open("constants.yaml", 'r') as stream:
        try:
            constants = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

    return constants

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

# =============================================================================
# Constants from constants.yaml
# =============================================================================

constants = read_constants()
TISSUE_DENSITY = constants["TISSUE_DENSITY"]
DOSE_CONSTANT = constants["DOSE_CONSTANT"]


