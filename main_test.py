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

    voxel_vol_ml: float

    def load_settings(self, path_settings):
        df = pd.read_csv(path_settings, sep=";", index_col=0)
        print("LOADED FROM settings.csv:", end="\t")
        self.time_to_img = float(df.loc["TIME TO IMG"].dropna().values)
        self.shunt_factor = float(df.loc["LSF"].dropna().values)

        # For plotting 3x3 100Gy regions, given various administerred activities (GBq)
        self.ind_window = [int(x) for x in df.loc["IND_WINDOW"].dropna().values]
        if "ACT_LEVELS" in df.index:
            self.act_levels = np.array([float(x) for x in df.loc["ACT_LEVELS"].dropna().values])
            self.act_levels = np.sort(self.act_levels)[::-1]    # countours want increasing levels (Gy-threshold decrease with increased activity)
        if "ADM_ACT" in df.index:
            act_vals = df.loc["ADM_ACT"].dropna().astype(float).values
            if len(act_vals) == 1:
                self.adm_act = float(act_vals)
                self.act_levels = []
            else:
                self.adm_act = np.sum(act_vals)
                self.act_levels = act_vals

            print(f"adm_act={self.adm_act} GBq ({self.act_levels})", end=", ")

        idx_segmentations = int(np.argwhere([ind[:3] == "***" for ind in df.index.values]))
        self.seg_operations = df.iloc[idx_segmentations + 1:, :]
        del df
        if self.segname_tot_counts:
            seg_names = list(set([*self.seg_operations.index, self.segname_tot_counts]))
        else:
            seg_names = list(set(self.seg_operations.index))

        self.seg_lookup = pd.DataFrame(index=seg_names,
                                       columns=["Layer", "Label"], dtype=int)

        print(f"time_to_img={self.time_to_img}, shunt_factor={self.shunt_factor}, "
              f"seg_operations with {len(self.seg_operations)} operations for {len(seg_names)} segmentations.")
        print(self.seg_operations)

        return 1


    def __init__(self, patient_top_dir: str, **kwargs):

        self.patient_top_dir = patient_top_dir
        self.voxel_vol_ml = 0 # TO TEST if changed in make_dosemap function
        self.CT_path = os.path.join(self.patient_top_dir, "CT.nrrd")
        self.SPECT_path = os.path.join(self.patient_top_dir, "SPECT.nrrd")
        self.PET_path = os.path.join(self.patient_top_dir, "PET.nrrd")

        segmentation_name = kwargs.get("segmentation_name", "Segmentation.seg.nrrd")
        self.segmentation_path = os.path.join(
            self.patient_top_dir, segmentation_name)

        self.segname_tot_counts = kwargs.get("segname_tot_counts", "counts_tot")

        self.dosemap_path = os.path.join(self.patient_top_dir, "dose_map.nrrd")

        # self.segmentation_label_map = os.path.join(
        #     self.patient_top_dir, "SegmentationLabelMap.nrrd")

        # LOAD self.time_to_img, self.shunt_factor, self.seg_table
        settings_name = kwargs.get("settings_name", "settings.csv")
        if settings_name:
            self.load_settings(os.path.join(self.patient_top_dir, settings_name))    # LFS, time-points, and seg_operations, seg_lookup, ind_window (for plotting)
        else:
            self.seg_lookup = pd.DataFrame(columns=["Layer", "Label"], dtype=int)
            print("settings.csv is required (for now)")
            sys.exit()

        # self.lookup_table_path = os.path.join(
        #     self.patient_top_dir, "Segmentation_1-label_ColorTable.ctbl")
        self.load_segmentations(path=self.segmentation_path, include_extras=True)


    def load_segmentations(self, path, include_extras=False):
        print("\nLOADING SEGMENTATIONS:", end="\t")
        self.seg, meta = nrrd.read(path, index_order="C")
        print(self.seg.shape)
        # print(meta)

        self.ndims_seg = meta["dimension"]
        if self.ndims_seg == 4:
            pass
        elif self.ndims_seg == 3:
            pass
        else:
            print("*** ERR: FOUND", self.ndims_seg, "DIMENSIONS IN", self.segmentation_path)
            sys.exit()


        seg_names_meta = list(filter(lambda k: "_Name" in k and "Auto" not in k, meta))
        seg_names_meta = {meta[nm]:nm.split("_")[0] for nm in seg_names_meta}
        # print(seg_names_meta)

        if self.segname_tot_counts != None:

            if not self.segname_tot_counts in seg_names_meta.keys():
                print("*** DID NOT FIND", self.segname_tot_counts, "IN SEGMENTATION...")
                sys.exit()
            else:
                # add counts_tot layer + label values to seg_table
                self.seg_lookup.loc[self.segname_tot_counts, ["Layer", "Label"]] = get_layer_label_values_from_meta(meta, segment=seg_names_meta[self.segname_tot_counts])
        else:
            print("\tsegname_tot_counts =", self.segname_tot_counts, "-> not relevant?")


        seg_overlap = set(seg_names_meta.keys()).intersection(self.seg_operations.index.values)

        print(f"\tFOUND {len(seg_overlap)} of {len(set(self.seg_operations.index))} segmentations from settings.csv", end="\t")
        # print(seg_overlap)

        seg_extras = set(seg_names_meta.keys()).difference(set(self.seg_operations.index.values))
        # print(seg_extras)

        if seg_extras:
            print(f"Found {len(seg_extras)} not in settings.csv:", seg_extras)
        else:
            print()
        if include_extras:
            seg_overlap = seg_overlap.union(seg_extras)

        for seg_nm in seg_overlap:
            self.seg_lookup.loc[seg_nm, ["Layer", "Label"]] = get_layer_label_values_from_meta(meta, segment=seg_names_meta[seg_nm])
        # print(self.seg_lookup)
        # sys.exit()
        self.seg_lookup = self.seg_lookup.astype(int)


        # Check if total counts segmentation contains all other segmentations as subsets, as is implied when using
        # fraction-map for dosimetry, excluding lungs (as lung counts are assumed included in the LSF)
        lungs_in_lookup = [idx for idx in self.seg_lookup.index if idx in ["Lungs", "LUNGS", "lungs"]]

        if self.segname_tot_counts == None:
            any_to_check = False
        else:
            any_to_check = any(self.seg_lookup.index.drop([*lungs_in_lookup, self.segname_tot_counts]))

        if any_to_check:

            if not check_segments_overlaps(self.seg, seg_tot_name=self.segname_tot_counts,
                                           df_lookup=self.seg_lookup.drop(lungs_in_lookup)):
                print(f"*** NON-OVERLAPPING VOXELS BETWEEN TOTAL COUNTS ({self.segname_tot_counts}) AND SEGMENTATIONS...")
            else:
                print("\tAll segments located in", self.segname_tot_counts, "-> ok (Lungs excluded)")
        else:
            print("\tNo segments contained in ", self.segname_tot_counts, " -> not necessary to check self-containment")

        print("\tADDED LAYER / LABEL TO seg_lookup for", len(self.seg_lookup), f"segmentations, added {len(self.seg_operations)} operations.")

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

    def get_segment_label_and_layer_from_lookup(self, seg_name):
        layer = self.seg_lookup.loc[seg_name, "Layer"]
        label = self.seg_lookup.loc[seg_name, "Label"]
        seg_layer = self.seg[:, :, :, layer]
        return label, seg_layer


    def calculate_XGy_region_volumes(self, dm, X=100, inside_segment=None):
        inside_segment = inside_segment if inside_segment else self.segname_tot_counts

        print(f"\nCALCULATING volumes for {X} Gy regions, inside {inside_segment}, when administerring {self.act_levels} GBq 90Y")
        gy_thresholds = np.array([X]*len(self.act_levels)) / self.act_levels
        print(f"\tcorresponding to Gy-thresholds on Gy / GBq voxel-map as:", np.round(gy_thresholds, 2))


        # Remove counts not in liver / segmentation used to calculate total counts
        # TODO: move to centralized loading? During dose-map creation? What other features not using (other) segmentations?

        # counts_total_label, seg_total = get_segment_label_and_layer_from_lookup(self.seg, inside_segment, self.seg_lookup)
        counts_total_label, seg_total = self.get_segment_label_and_layer_from_lookup(seg_name=inside_segment)
        # seg_total[seg_total == counts_total_label] = 1
        # seg_total[seg_total != counts_total_label] = 0

        dm_in_tot = dm[seg_total == counts_total_label]

        print(f"\tDOSEMAP voxel volume (cm3) = {self.voxel_vol_ml:.3f}", dm.shape)
        for act, gy in zip(self.act_levels, gy_thresholds):
            # ALL Gy's ARE PER GBq ADMINISTERRED (IMPORTANT TO MULTIPLY BY act FOR ACTUAL DOSE CALCULATIONS..!!!)

            num_vx = len(dm_in_tot[dm_in_tot >= gy])
            vol = self.voxel_vol_ml * num_vx
            print(f"\t{act} GBq ({gy:.1f} Gy threshold @ Gy / GBq dosemap) -> {vol:.3f} cm3")

            for seg_name in self.seg_operations.index:
                # sg_label, seg_sg = get_segment_label_and_layer_from_lookup(self.seg, sg, self.seg_lookup)
                seg_label, seg = self.get_segment_label_and_layer_from_lookup(seg_name)
                dm_in_seg = dm[seg == seg_label]
                num_vx_seg = len(dm_in_seg[dm_in_seg >= gy])
                mean_in_seg = np.mean(dm_in_seg[dm_in_seg >= gy]) * act
                vol_seg = self.voxel_vol_ml * num_vx_seg
                print(f"\t\tInside {seg_name}:\t\t{vol_seg:.2f} cm3 (mean = {mean_in_seg:.2f} Gy)")

        pass


    def plot_XGy_regions(self, X=100):

        # Use meta_spect to interpolate dm to CT-shape?
        # NOTE: sitk cannot handle reading norwegian letters in file-name (ÆØÅ)

        import SimpleITK as sitk
        print(self.CT_path)
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
            gy_thresholds = np.array([X, X, X]) / self.act_levels
            # levels = np.sort(levels)
            colors = ["cyan", "yellow", "red"]
            print(self.act_levels)
            print(gy_thresholds)

            dm_slice = DM_RESAMP[ind, :, :]
            # plt.imshow(dm_slice, alpha=0.50, cmap="hot")
            ax[i].contour(dm_slice, levels=gy_thresholds, colors=colors)
            ax[i].axis("off")

        # fig.tight_layout()
        pad = -.1
        fig.subplots_adjust(wspace=pad, hspace=pad)
        fig.suptitle(f"{X}Gy regions for {''', '''.join(self.act_levels.astype(str))} GBq administerred 90Y")
        plt.show()


        pass


    def make_dosemap_decay_corrected(self, reference_geometry = "PET"):
        decay_factor = np.exp(+(np.log(2) / sirt.HALF_LIFE_HRS) * self.time_to_img)
        print("\n CREATING dosemap using", reference_geometry, end="\t")

        if reference_geometry == "PET":
            input_path = self.PET_path
        else:
            logger.error("Reference geometry not recognized")
            raise ValueError("Reference geometry not recognized")

        input_array, input_header = nrrd.read(input_path, index_order="C")
        print(input_array.shape)
        print(f"\tDecay factor = {decay_factor:.2f}")
        label_array = self.seg


        # total_mask = np.zeros(input_array.shape)
        # assert input_array.shape == label_array.shape[:-1] == total_mask.shape

        space_dirs = (input_header["space directions"])

        x_dim = np.abs((space_dirs[0, 0])) / 10 ## mm -> cm
        y_dim = np.abs((space_dirs[1, 1])) / 10
        z_dim = np.abs((space_dirs[2, 2])) / 10

        self.voxel_vol_ml = x_dim * y_dim * z_dim
        total_volume = np.prod(input_array.shape) * self.voxel_vol_ml
        anisotropy_threshold = 0.006

        if not (abs(x_dim - y_dim) < anisotropy_threshold) or not (abs(y_dim - z_dim) < anisotropy_threshold):

            print(f"*** ANISOTROPE VOXELS... (x={x_dim}, y={y_dim}, z={z_dim})")
            sys.exit()
        else:
            d = x_dim * 10  # cm -> mm
            s_scaling_factor = sirt.VOX_DIMS_ORIG ** 3 / d ** 3
            # print(d, s_scaling_factor)
            print(f"\tS-FACTOR SCALING 4.42^3 / d^3 = {s_scaling_factor:.3f}, using d={d:.2f} mm voxel sizes")


        activity_map = input_array * decay_factor * self.voxel_vol_ml
        activity_map *= 1e-9    # Bq to GBq

        if self.segname_tot_counts == None:
            sum_act = np.sum(activity_map)
            print(f"\tSum of activity in whole image (segname_tot_count={self.segname_tot_counts} --> no LSF correction) = "
                  f"{sum_act:.3e} Bq ({sum_act / (self.adm_act)*100:.1f} % of {self.adm_act} GBq administerred)")

        else:
            counts_total_index, counts_total_layer = self.get_segment_label_and_layer_from_lookup(self.segname_tot_counts)
            print("\t", counts_total_layer.shape, input_data.seg.shape)
            sum_act = np.sum(activity_map[counts_total_layer == counts_total_index]) / (1-self.shunt_factor)

            print(f"\tSum of activity in {self.segname_tot_counts} ({self.shunt_factor*100:.0f}% LSF-corrected) = "
                  f"{sum_act:.3e} Bq ({sum_act / (self.adm_act)*100:.1f} % of {self.adm_act} GBq administerred)")


        dose_map = activity_map * np.log(2)**-1 * sirt.HALF_LIFE_SEC * sirt.S_FACTOR * s_scaling_factor

        for seg_nm in self.seg_lookup.index.values:
            # counts_total_index, counts_total_layer = get_segment_label_and_layer_from_lookup(input_data.seg, seg_nm, input_data.seg_lookup)
            counts_seg_index, counts_seg_layer = self.get_segment_label_and_layer_from_lookup(seg_nm)
            seg_data = input_array[counts_seg_layer == counts_seg_index]
            # print(seg_data.shape, counts_seg_layer.shape)

            # counts_seg = np.sum(seg_data)
            num_vox_seg = len(seg_data)
            vol_seg = input_data.voxel_vol_ml * num_vox_seg

            act_seg = np.sum(activity_map[counts_seg_layer == counts_seg_index]) / (1-self.shunt_factor)

            dose_seg = dose_map[counts_seg_layer == counts_seg_index]

            print(f"\t{seg_nm} ({vol_seg:.2f} mL):\t\ttotal activity = {act_seg*1e3:.2g} MBq ({act_seg / self.adm_act * 100:.1f}% of {self.adm_act} GBq adm)"
                  f"\tmean / median dose {np.mean(dose_seg):.2f} / {np.median(dose_seg):.2f} Gy")

        nrrd.write(input_data.dosemap_path, dose_map, header=input_header, index_order='C')
        print("\tDOSEMAP saved as:", input_data.dosemap_path)
        return dose_map


def make_dosemap(input_data: InputDataSIRT, shunt_factor: float = 0.0, reference_geometry = "SPECT"):

    print("\nCREATING dosemap using", reference_geometry, end="\t")

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

    input_data.voxel_vol_ml = x_dim * y_dim * z_dim

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

    counts_total_index, counts_total_layer = input_data.get_segment_label_and_layer_from_lookup(input_data.segname_tot_counts)

    # print(input_data.seg_operations)
    print("\t", counts_total_layer.shape, input_data.seg.shape)
    # sys.exit()


    # tumor_dict = input_data.segmentation.tumor_index_dict

    # total_mask[label_array == counts_total_index] = 1
    # total_mask[counts_total_layer == counts_total_index] = 1
    # print(total_mask)

    # sys.exit()

    # for tum in tumor_dict.keys():

        # total_mask[label_array == tumor_dict[tum]] = 1

    # total = np.sum(input_array[total_mask == 1])
    total_counts = np.sum(input_array[counts_total_layer == counts_total_index])

    num_vox_in_total_counts = len(input_array[counts_total_layer == counts_total_index])
    num_vox_in_whole_image = np.prod(input_array.shape)
    vol_total_counts = num_vox_in_total_counts * input_data.voxel_vol_ml

    print("\tTOTAL COUNTS:", total_counts, f" = {total_counts / np.sum(input_array)*100:.1f}% of counts in whole image, in {num_vox_in_total_counts / num_vox_in_whole_image * 100:.1f} % of voxels ({vol_total_counts:.2f} mL)")


    total_counts = total_counts / (1 - shunt_factor)

    image_as_double = input_array.astype(np.float32)

    fraction_image = image_as_double / total_counts

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

    # for tum in tumor_dict.keys():
    for seg_nm in input_data.seg_lookup.index.values:
        # counts_total_index, counts_total_layer = get_segment_label_and_layer_from_lookup(input_data.seg, seg_nm, input_data.seg_lookup)
        counts_seg_index, counts_seg_layer = input_data.get_segment_label_and_layer_from_lookup(seg_nm)
        seg_data = input_array[counts_seg_layer == counts_seg_index]

        seg_total = np.sum(seg_data)
        num_vox_seg = len(seg_data)
        vol_seg = input_data.voxel_vol_ml * num_vox_seg

        dose_seg = dose_map[counts_seg_layer == counts_seg_index]

        print(f"\t{seg_nm} ({vol_seg:.2f} mL): {seg_total/total_counts * 100:.1f}% of used counts, with mean dose {np.mean(dose_seg):.2f} Gy / GBq")

    # sys.exit()

    # sys.exit()
    # nrrd.write("dose_map.nrrd", dose_map, header=input_header, index_order='C')
    # dosemap_path = os.path.join(input_data.patient_top_dir, "dose_map.nrrd")
    nrrd.write(input_data.dosemap_path, dose_map, header=input_header, index_order='C')
    print("\tDOSEMAP saved as:", input_data.dosemap_path)
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

# patient_top_dir = r"C:\Users\toral\OneDrive\OUS\SIRT_dev\BS50\work-up"
# patient_top_dir = r"E:\SIRT\EKR52\Slicer"
# patient_top_dir = r"E:\SIRT\AAMSW82\Slicer"
# patient_top_dir = r"E:\SIRT\AAMSW82\verif"
patient_top_dir = r"E:\SIRT\EKR52\verif"


input_data = InputDataSIRT(patient_top_dir=patient_top_dir,
                           # segname_tot_counts="Liver",
                           segname_tot_counts=None,
                           # segname_tot_counts="TotalCounts",
                           # segmentation_name="dosemap_segmentation.seg.nrrd",
                           segmentation_name="Segmentation.seg.nrrd",
                           settings_name="settings.csv")

# input_data.check_files()  # TODO: do this somewhere

# alias_dict = input_data.segmentation.tumor_alias_dict

# dose_map = make_dosemap(input_data, shunt_factor=input_data.shunt_factor, reference_geometry="SPECT")
dose_map = input_data.make_dosemap_decay_corrected()
sys.exit()

# input_data.act_levels = [1.6]
input_data.act_levels = [1.0]

input_data.calculate_XGy_region_volumes(dose_map)
sys.exit()

input_data.plot_XGy_regions()


# calculate_bq_for_segmentations_operations(dose_map, input_data)
# input_data.seg_operations


segment_voxels = voxel_values_by_segment_name(input_data, dose_map, "Tum1")
make_cDVH(segment_voxels, )

# print(alias_dict)

# sys.exit()


#input_data.check_files()

sys.exit()
