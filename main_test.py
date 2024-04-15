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
    seg_operations: pd.DataFrame    # operations to perform on segmentations (multiple OK)
    seg_lookup: pd.DataFrame        # lookup-table for layer, value for segmentation names (one entry per segment name)

    seg: np.array

    def load_settings(self, path_settings):
        df = pd.read_csv(path_settings, sep=";", index_col=0)
        self.time_to_img = float(df.loc["TIME TO IMG"].dropna().values)
        self.shunt_factor = float(df.loc["LSF"].dropna().values)

        # For plotting 3x3 100Gy regions, given various administerred activities (GBq)
        self.ind_window = [int(x) for x in df.loc["IND_WINDOW"].dropna().values]
        self.act_levels = np.array([float(x) for x in df.loc["ACT_LEVELS"].dropna().values])

        idx_segmentations = int(np.argwhere([ind[:3] == "***" for ind in df.index.values]))
        self.seg_operations = df.iloc[idx_segmentations + 1:, :]
        del df
        seg_names = set(self.seg_operations.index)
        self.seg_lookup = pd.DataFrame(index=[*list(seg_names), self.segname_tot_counts],
                                       columns=["Layer", "Label"], dtype=int)

        print(f"LOADED FROM settings.csv: time_to_img={self.time_to_img}, shunt_factor={self.shunt_factor}, "
              f"seg_operations with {len(self.seg_operations)} operations for {len(seg_names)} segmentations.")
        print(self.seg_operations)

        return 1


    def __init__(self, patient_top_dir: str, **kwargs):

        self.patient_top_dir = patient_top_dir

        self.CT_path = os.path.join(self.patient_top_dir, "CT.nrrd")
        self.SPECT_path = os.path.join(self.patient_top_dir, "SPECT.nrrd")
        self.segmentation_path = os.path.join(
            self.patient_top_dir, "Segmentation.seg.nrrd")
        self.segname_tot_counts = kwargs.get("segname_tot_counts", "counts_tot")

        self.dosemap_path = os.path.join(self.patient_top_dir, "dose_map.nrrd")

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
            print("*** DID NOT FIND", self.segname_tot_counts, "IN SEGMENTATION...")
            sys.exit()
        else:
            # add counts_tot layer + label values to seg_table
            self.seg_lookup.loc[self.segname_tot_counts, ["Layer", "Label"]] = get_layer_label_values_from_meta(meta, segment=seg_names_meta[self.segname_tot_counts])

        seg_overlap = set(seg_names_meta.keys()).intersection(self.seg_operations.index.values)
        # print(seg_overlap)
        # print(self.seg_table.index.values)

        print(f"\tFOUND {len(seg_overlap)} of {len(set(self.seg_operations.index))} segmentations from settings.csv", end="\t")

        seg_extras = set(seg_names_meta.keys()).difference(set(self.seg_operations.index.values))
        # print(seg_extras)

        if seg_extras:
            print(f"BUT found {len(seg_extras)} not in settings.csv:", seg_extras)
        else:
            print()

        for seg_nm in seg_overlap:
            self.seg_lookup.loc[seg_nm, ["Layer", "Label"]] = get_layer_label_values_from_meta(meta, segment=seg_names_meta[seg_nm])
        self.seg_lookup = self.seg_lookup.astype(int)

        if not check_segments_overlaps(self.seg, seg_tot_name=self.segname_tot_counts,
                                       df_lookup=self.seg_lookup.drop("Lungs")):
            print(f"*** NON-OVERLAPPING VOXELS BETWEEN TOTAL COUNTS ({self.segname_tot_counts}) AND SEGMENTATIONS...")
        else:
            print("\tAll segments located in", self.segname_tot_counts, "-> ok (Lungs excluded)")

        # sys.exit()
        print("\tADDED LAYER / LABEL TO seg_lookup")

        # print(self.seg_operations)
        # print(self.seg_lookup)

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

    def make_XGy_regions(self, X=100):
        # Use meta_spect to interpolate dm to CT-shape?
        import SimpleITK as sitk

        ct = sitk.ReadImage(self.CT_path)
        spect = sitk.ReadImage(self.SPECT_path)
        dm = sitk.ReadImage(self.dosemap_path)

        ct_size, ct_spacing = ct.GetSize(), ct.GetSpacing()
        # print(ct_size, ct_spacing)
        # print(dm.GetSize())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetSize(ct_size)
        resampler.SetOutputSpacing(ct_spacing)
        resampler.SetTransform(sitk.AffineTransform(3))

        # spect_resamp = resampler.Execute(spect)
        dm_resamp = resampler.Execute(dm)

        CT = sitk.GetArrayFromImage(ct)
        SPECT = sitk.GetArrayFromImage(spect)
        DM_RESAMP = sitk.GetArrayFromImage(dm_resamp)
        # print(CT.shape, DM_RESAMP.shape, np.mean(DM_RESAMP), np.mean(sitk.GetArrayFromImage(dm)))

        # print(self.ind_window)  # on SPECT
        ind_window_ct = np.array(self.ind_window) * CT.shape[0] / SPECT.shape[0]
        ind_window_ct = ind_window_ct
        indices = np.linspace(ind_window_ct[0], ind_window_ct[1], 9).astype(int)


        print(self.act_levels)
        # ind = 35
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
        ax = ax.ravel()
        for i, ind in enumerate(indices):
            ax[i].imshow(CT[ind, :, :], alpha=1, vmin=-135, vmax=215, cmap="gray")
            # plt.imshow(SPECT_RESAMP[ind, :, :], alpha=0.50, cmap="hot")
            # dm_ma = np.ma.masked_where(np.logical_not(DM_RESAMP[ind, :, :]), DM_RESAMP[ind, :, :])

            # plt.imshow(dm_ma, alpha=0.50, cmap="hot")
            levels = np.array([X, X, X]) / self.act_levels
            # levels = np.sort(levels)
            colors = ["cyan", "yellow", "red"]
            print(self.act_levels)
            print(levels)

            dm_slice = DM_RESAMP[ind, :, :]
            # plt.imshow(dm_slice, alpha=0.50, cmap="hot")
            ax[i].contour(dm_slice, levels=levels, colors=colors)
            ax[i].axis("off")

        # fig.tight_layout()
        pad = -.1
        fig.subplots_adjust(wspace=pad, hspace=pad)
        fig.suptitle(f"{X}Gy regions for {''', '''.join(self.act_levels.astype(str))} GBq administerred 90Y")
        plt.show()


        pass


def make_dosemap(input_data: InputDataSIRT, shunt_factor: float = 0.0, reference_geometry = "SPECT"):

    print("CREATING dosemap using", reference_geometry, end="\t")

    if reference_geometry == "SPECT":
        input_path = input_data.SPECT_path
    
    else:

        logger.error("Reference geometry not recognized")
        raise ValueError("Reference geometry not recognized")
    
    if shunt_factor == 0.0:
        logger.warning("Shunt factor is 0.0 - is this correct?")

    # Use the total and the tumor to calculate the total

    input_array, input_header = nrrd.read(input_path, index_order='C')  # SPECT
    # label_array, label_header = nrrd.read(input_data.segmentation_label_map, index_order='C')
    print(input_array.shape)
    label_array = input_data.seg

    total_mask = np.zeros(input_array.shape)

    # assert input_array.shape == label_array.shape == total_mask.shape
    assert input_array.shape == label_array.shape[:-1] == total_mask.shape
    # Check the spacing

    space_dirs = (input_header["space directions"])

    x_dim = np.abs((space_dirs[0, 0])) / 10
    y_dim = np.abs((space_dirs[1, 1])) / 10
    z_dim = np.abs((space_dirs[2, 2])) / 10

    if not (round(x_dim, 2) == round(y_dim, 2) and round(y_dim, 2) == round(z_dim, 2)):
        print("*** ANISOTROPE VOXELS...")
        sys.exit()
    else:
        d = x_dim * 10  # cm -> mm
        s_scaling_factor = sirt.VOX_DIMS_ORIG**3 / d**3
        # print(d, s_scaling_factor)
        print(f"\tS-FACTOR SCALING d^3 / 4.42^3 = {s_scaling_factor:.3f}, using d={d:.2f} mm voxel sizes")


    voxel_mass = (x_dim * y_dim * z_dim * sirt.TISSUE_DENSITY) / 1e3  # Voxel mass in kg

    # liver_total_index = input_data.segmentation.total_count_index
    # liver_total_index, liver_total_layer = input_data.seg_operations.loc[input_data.segname_tot_counts, ["Label", "Layer"]]

    counts_total_index, counts_total_layer = get_segment_label_and_layer_from_lookup(input_data.seg, input_data.segname_tot_counts, input_data.seg_lookup)

    # print(input_data.seg_operations)
    print(counts_total_layer.shape)


    # tumor_dict = input_data.segmentation.tumor_index_dict

    # total_mask[label_array == counts_total_index] = 1
    # total_mask[counts_total_layer == counts_total_index] = 1
    # print(total_mask)

    # sys.exit()

    # for tum in tumor_dict.keys():

        # total_mask[label_array == tumor_dict[tum]] = 1

    # total = np.sum(input_array[total_mask == 1])
    total = np.sum(input_array[counts_total_layer == counts_total_index])
    print("\tTOTAL COUNTS:", total, f"({total / np.sum(input_array)*100:.1f}% of whole image)")

    # for tum in tumor_dict.keys():
    for seg_nm in input_data.seg_lookup.index.values:
        counts_total_index, counts_total_layer = get_segment_label_and_layer_from_lookup(input_data.seg, seg_nm, input_data.seg_lookup)
        seg_total = np.sum(input_array[counts_total_layer == counts_total_index])
        # tum_total = np.sum(input_array[label_array == tumor_dict[tum]])
        print("\t", seg_nm, round(seg_total/total, 3))


    total = total / (1 - shunt_factor)

    image_as_double = input_array.astype(np.float32)

    fraction_image = image_as_double / total

    assert fraction_image.shape == input_array.shape
    assert np.max(fraction_image) <= 1.0
    assert np.min(fraction_image) >= 0.0

    print("\tSum fraction image:", np.sum(fraction_image))
    print(f"\t\twithin {input_data.segname_tot_counts}:", np.sum(fraction_image[counts_total_layer == counts_total_index]))

    # dose_map = (fraction_image * sirt.DOSE_CONSTANT) / (voxel_mass)
    # print(np.mean(dose_map))
    # mean_old = np.mean(dose_map)

    dose_factor = np.log(2)**-1 * sirt.HALF_LIFE_SEC * sirt.S_FACTOR * s_scaling_factor
    dose_map = fraction_image * dose_factor
    # print(np.mean(dose_map), np.mean(dose_map) / mean_old)

    # sys.exit()
    # nrrd.write("dose_map.nrrd", dose_map, header=input_header, index_order='C')
    # dosemap_path = os.path.join(input_data.patient_top_dir, "dose_map.nrrd")
    nrrd.write(input_data.dosemap_path, dose_map, header=input_header, index_order='C')

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

# dose_map = make_dosemap(input_data, shunt_factor=0.0, reference_geometry="SPECT")
dose_map = make_dosemap(input_data, shunt_factor=input_data.shunt_factor, reference_geometry="SPECT")

input_data.make_XGy_regions()

# calculate_bq_for_segmentations_operations(dose_map, input_data)
# input_data.seg_operations

sys.exit()

segment_voxels = voxel_values_by_segment_name(input_data, dose_map, "Tum1")
make_cDVH(segment_voxels, )

# print(alias_dict)

# sys.exit()


#input_data.check_files()

sys.exit()
