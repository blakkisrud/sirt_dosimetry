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
import logging
import os

# =============================================================================
# Run constants - could be read from the yaml-file
# =============================================================================

DO_DELETE_LOG_FILE = False

# =============================================================================
# Functions
# =============================================================================

def setup_logger(f_name = "program.log"):

    """
    Set up the logger

    Returns
    -------
    logger : logging.Logger
        The logger object
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler for logger

    fh = logging.FileHandler(f_name)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logger set up")

    return logger

def delete_log_file(f_name = "program.log"):
    """
    Delete the log file
    """

    if os.path.exists(f_name):
        os.remove(f_name)
        print("File deleted: ", f_name)
    else:
        print("The file does not exist")

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
    # sum_of_counts = 8580571.42857143

    fraction_image = image_as_double / sum_of_counts

    # Calculate and save dose map
    # Re-use the image header from the count-image

    dose_map = (fraction_image * DOSE_CONSTANT) / (voxel_mass);
    dose_factor = DOSE_CONSTANT/voxel_mass
    print("\tdose_factor? = ", dose_factor)
    if output_path:
        nrrd.write(output_path, dose_map, header=image_header, index_order='C')
        print("Dose map saved as nrrd at", output_path)
    return dose_map



# =============================================================================
# Constants from constants.yaml
# =============================================================================

constants = read_constants()
TISSUE_DENSITY = constants["TISSUE_DENSITY"]
DOSE_CONSTANT = constants["DOSE_CONSTANTS"]
S_FACTOR = constants["S_FACTOR"]
VOX_DIMS_ORIG = constants["VOX_DIMS_ORIG"]
HALF_LIFE_HRS = constants["HALF_LIFE_HRS"]
HALF_LIFE_SEC = HALF_LIFE_HRS * 3600

# =============================================================================
# Logger
# =============================================================================

if DO_DELETE_LOG_FILE:
    delete_log_file()
    logger = setup_logger()

else:

    logger = setup_logger()



