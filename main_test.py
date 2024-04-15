"""
Script to test functions for the main SIRT application

When tests are done and satisfactory, the functions can be used in the main
file source code.

"""
from utils import *
import sirt_functions as sirt
import os
import pandas as pd
# import vtk
import xml.etree.ElementTree as ET
import SimpleITK as sitk
import numpy as np
import nrrd
# import slicer
import sys
from dataclasses import dataclass
import matplotlib.pyplot as plt
import yaml


@dataclass
class Segmentations:

    patient_top_dir: str


    # total_count_index: int  # This is exluding tumors
    # lung_index: int
    # num_of_tumors: int  # Might be superfluous
    # tumor_index_dict: dict
    # tumor_alias_dict: dict


    def __init__(self, patient_top_dir: str):

        self.patient_top_dir = patient_top_dir

        # self.load_segmentations()


        # lookup_table_path = os.path.join(
        #     self.patient_top_dir, "Segmentation_1-label_ColorTable.ctbl")
        
        # self.lookup_table = self.load_loopup_table(lookup_table_path)

        # print("Lookup table: ", self.lookup_table)

        # Check if "LiverTotal" is in index of dict

        if "LiverTotal" in self.lookup_table:
            self.total_count_index = self.lookup_table["LiverTotal"]

        if "Lung" in self.lookup_table:
            self.lung_index = self.lookup_table["Lung"]

        else:
            logger.warning("Lung not in lookup table - is this correct?")

        tumor_list = [key for key in self.lookup_table.keys() if "Tum" in key]

        self.num_of_tumors = len(tumor_list)

        self.tumor_index_dict = {}

        for tum in tumor_list:

            self.tumor_index_dict[tum] = self.lookup_table[tum]

        input_data_yaml = os.path.join(self.patient_top_dir, "input.yaml")

        with open(input_data_yaml, 'r') as f:
                
            try:
    
                input_data_dict = yaml.safe_load(f)
    
            except yaml.YAMLError as exc:
    
                logger.error(exc)
                raise exc

        self.tumor_alias_dict = {} 

        for key in input_data_dict.keys():

            if "Tum" in key:

                self.tumor_alias_dict[key.replace("_alias", "")] = input_data_dict[key]
                print(key)
    



@dataclass
class InputDataSIRT:

    patient_top_dir: str

    CT_path: str
    SPECT_path: str
    segmentation_path: str
    # segmentation_label_map: str
    # lookup_table_path: str

    # input_data_yaml: str

    time_to_img: float
    shunt_factor: float
    seg_table: pd.DataFrame

    seg: np.array

    def load_settings(self, path_settings):
        df = pd.read_csv(path_settings, sep=";", index_col=0)
        self.time_to_img = float(df.loc["TIME TO IMG"].dropna().values)
        self.shunt_factor = float(df.loc["LSF"].dropna().values)


        idx_segmentations = int(np.argwhere([ind[:3] == "***" for ind in df.index.values]))
        self.seg_table = df.iloc[idx_segmentations+1:, :]
        del df

        print(f"LOADED FROM settings.csv: time_to_img={self.time_to_img}, shunt_factor={self.shunt_factor}, "
              f"seg_table with {len(self.seg_table)} segmentations:")
        print(self.seg_table)
        return 1


    def __init__(self, patient_top_dir: str, **kwargs):

        self.patient_top_dir = patient_top_dir

        self.CT_path = os.path.join(self.patient_top_dir, "CT.nrrd")
        self.SPECT_path = os.path.join(self.patient_top_dir, "SPECT.nrrd")
        self.segmentation_path = os.path.join(
            self.patient_top_dir, "Segmentation.seg.nrrd")
        self.segname_tot_counts = kwargs.get("segname_tot_counts", "counts_tot")

        # self.segmentation_label_map = os.path.join(
        #     self.patient_top_dir, "SegmentationLabelMap.nrrd")

        # LOAD self.time_to_img, self.shunt_factor, self.seg_table
        self.load_settings(os.path.join(self.patient_top_dir, "settings.csv"))    # LFS, time-points, and segmentation_table

        # self.lookup_table_path = os.path.join(
        #     self.patient_top_dir, "Segmentation_1-label_ColorTable.ctbl")

        self.load_segmentations(path=self.segmentation_path)



    def load_segmentations(self, path):
        print("LOADING SEGMENTATIONS:", end="\t")
        self.seg, meta = nrrd.read(path, index_order="C")
        print(self.seg.shape)
        # print(meta)

        if meta["dimension"] != 4:
            print("NOT IMPLEMENTED 3-DIM SEGMENTATION INPUT")
            sys.exit()

        seg_names_meta = list(filter(lambda k: "_Name" in k and "Auto" not in k, meta))
        seg_names_meta = {meta[nm]:nm.split("_")[0] for nm in seg_names_meta}
        # print(seg_names_meta)

        if not self.segname_tot_counts in seg_names_meta.keys():
            print("DID NOT FIND", self.segname_tot_counts, "IN SEGMENTATION...")
            sys.exit()
        else:
            # add counts_tot layer + label values to seg_table
            self.seg_table.loc[self.segname_tot_counts, ["Layer", "Label"]] = get_layer_label_values_from_meta(meta, segment=seg_names_meta[self.segname_tot_counts])


        seg_overlap = set(seg_names_meta.keys()).intersection(self.seg_table.index.values)
        # print(seg_overlap)
        # print(self.seg_table.index.values)

        print(f"\tFOUND {len(seg_overlap)} of {len(set(self.seg_table.index))} segmentations from settings.csv", end="\t")

        seg_extras = set(seg_names_meta.keys()).difference(set(self.seg_table.index.values))
        # print(seg_extras)

        if seg_extras:
            print(f"BUT found {len(seg_extras)} not in settings.csv:", seg_extras)
        else:
            print()

        for seg_nm in seg_overlap:
            self.seg_table.loc[seg_nm, ["Layer", "Label"]] = get_layer_label_values_from_meta(meta, segment=seg_names_meta[seg_nm])
        print("\tADDED LAYER / LABEL TO seg_table")
        # print(self.seg_table)

        pass

    def check_files(self):

        if not os.path.exists(self.CT_path):
            logger.error(f"CT-file {self.CT_path} does not exist")
            raise FileNotFoundError(f"CT-file {self.CT_path} does not exist")

        if not os.path.exists(self.SPECT_path):
            logger.error(f"SPECT-file {self.SPECT_path} does not exist")
            raise FileNotFoundError(f"SPECT-file {self.SPECT_path} does not exist")

        if not os.path.exists(self.segmentation_path):
            logger.error(
                f"Segmentation file {self.segmentation_path} does not exist")
            raise FileNotFoundError(
                f"Segmentation file {self.segmentation_path} does not exist")

        if not os.path.exists(self.segmentation_label_map):
            logger.error(
                f"Segmentation binary file {self.segmentation_label_map} does not exist")
            raise FileNotFoundError(
                f"Segmentation binary file {self.segmentation_label_map} does not exist")

        if not os.path.exists(self.lookup_table_path):
            logger.error(
                f"Lookup table {self.lookup_table_path} does not exist")
            raise FileNotFoundError(
                f"Lookup table {self.lookup_table_path} does not exist")

        if not os.path.exists(self.input_data_yaml):
            logger.error(
                f"Input data yaml file {self.input_data_yaml} does not exist")
            raise FileNotFoundError(
                f"Input data yaml file {self.input_data_yaml} does not exist")

        logger.info("All files exist")



def make_dosemap(input_data: InputDataSIRT, shunt_factor: float = 0.0, reference_geometry = "SPECT"):

    if reference_geometry == "SPECT":
        input_path = input_data.SPECT_path
    
    else:

        logger.error("Reference geometry not recognized")
        raise ValueError("Reference geometry not recognized")
    
    if shunt_factor == 0.0:
        logger.warning("Shunt factor is 0.0 - is this correct?")

    # Use the total and the tumor to calculate the total
        
    input_array, input_header = nrrd.read(input_path, index_order='C')  # SPECT
    label_array, label_header = nrrd.read(input_data.segmentation_label_map, index_order='C')

    total_mask = np.zeros(input_array.shape)

    assert input_array.shape == label_array.shape == total_mask.shape

    # Check the spacing

    space_dirs = (input_header["space directions"])

    x_dim = np.abs((space_dirs[0, 0])) / 10
    y_dim = np.abs((space_dirs[1, 1])) / 10
    z_dim = np.abs((space_dirs[2, 2])) / 10

    voxel_mass = (x_dim * y_dim * z_dim * sirt.TISSUE_DENSITY) / 1e3  # Voxel mass in kg

    liver_total_index = input_data.segmentation.total_count_index

    tumor_dict = input_data.segmentation.tumor_index_dict

    total_mask[label_array == liver_total_index] = 1

    for tum in tumor_dict.keys():
            
        total_mask[label_array == tumor_dict[tum]] = 1

    total = np.sum(input_array[total_mask == 1])

    for tum in tumor_dict.keys():

        tum_total = np.sum(input_array[label_array == tumor_dict[tum]])
        print(tum_total/total)

    total = total / (1 - shunt_factor)

    image_as_double = input_array.astype(np.float32)

    fraction_image = image_as_double / total

    assert fraction_image.shape == input_array.shape
    assert np.max(fraction_image) <= 1.0
    assert np.min(fraction_image) >= 0.0

    print(np.sum(fraction_image))

    dose_map = (fraction_image * sirt.DOSE_CONSTANT) / (voxel_mass)

    nrrd.write("dose_map.nrrd", dose_map, header=input_header, index_order='C')

    return dose_map

def make_cDVH(list_of_voxels, units = "Gy", dosage_levels = [1.0, 1.5, 2.0], tum_name = None):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if tum_name:
        ax.set_title(f"cDVH for tumor {tum_name}")

    for act in dosage_levels:

        dose_act = list_of_voxels * act

        D_range = np.linspace(0, np.max(dose_act), 100)
        volume_fractions = np.zeros(len(D_range))

        total_num_voxels = float(len(list_of_voxels))

        for D, i in zip(D_range, range(len(D_range))):

            volume_fractions[i] = len(dose_act[dose_act > D]) / total_num_voxels

        ax.plot(D_range, volume_fractions, label=f"{act} GBq")

    ax.set_xlabel(f"Dose ({units})")
    ax.set_ylabel("Volume fraction")

    ax.legend()

    plt.show()

def voxel_values_by_segment_name(input_data: InputDataSIRT, dose_map, segment_name: str):

    label_array, label_header = nrrd.read(input_data.segmentation_label_map, index_order='C')

    segment_index = input_data.segmentation.lookup_table[segment_name]

    segment_voxels = dose_map[label_array == segment_index]

    return segment_voxels

logger = sirt.logger

logger.info("Testing functions")

patient_top_dir = r"C:\Users\toral\OneDrive\OUS\SIRT_dev\BS50\work-up"

input_data = InputDataSIRT(patient_top_dir=patient_top_dir)
# input_data.check_files()  # TODO: do this somewhere

# alias_dict = input_data.segmentation.tumor_alias_dict

dose_map = make_dosemap(input_data, shunt_factor=0.0, reference_geometry="SPECT")
sys.exit()

segment_voxels = voxel_values_by_segment_name(input_data, dose_map, "Tum1")
make_cDVH(segment_voxels, )

# print(alias_dict)

sys.exit()





#input_data.check_files()

sys.exit()
