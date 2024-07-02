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
        if segmentation_name:
            self.segmentation_path = os.path.join(
                self.patient_top_dir, segmentation_name)
        else:
            self.segmentation_path = None

        self.segname_tot_counts = kwargs.get("segname_tot_counts", "counts_tot")

        self.dosemap_path = os.path.join(self.patient_top_dir, "dose_map.nrrd")

        # self.segmentation_label_map = os.path.join(
        #     self.patient_top_dir, "SegmentationLabelMap.nrrd")

        # LOAD self.time_to_img, self.shunt_factor, self.seg_table
        settings_name = kwargs.get("settings_name", "settings.csv")
        if settings_name:
            self.load_settings(os.path.join(self.patient_top_dir, settings_name))    # LFS, time-points, and seg_operations, seg_lookup, ind_window (for plotting)
            self.load_segmentations(path=self.segmentation_path, include_extras=True)

        else:
            print("\tNO SETTINGS FOUND: assuming dt = 0 hrs, shunt_factor = 0, adm_act = 1 GBq -> OVERWRITE MANUALLY AFTER INIT")
            # self.seg = np.zeros(shape=)
            self.seg = None
            self.seg_lookup = pd.DataFrame()
            self.time_to_img = 0
            self.shunt_factor = 0
            self.adm_act = 1
            # self.seg_lookup = pd.DataFrame(columns=["Layer", "Label"], dtype=int)
            # print("settings.csv is required (for now)")
            # sys.exit()
            pass


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
        try:
            self.seg_lookup = self.seg_lookup.astype(int)
        except Exception as e:
            print(*e.args)

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

    def get_segment_label_and_layer_from_lookup(self, seg_name, return_layer_as_int=False):
        layer = self.seg_lookup.loc[seg_name, "Layer"]
        label = self.seg_lookup.loc[seg_name, "Label"]
        if self.seg.ndim == 4:
            seg_layer = self.seg[:, :, :, int(layer)]
        else:
            seg_layer = self.seg
        if return_layer_as_int:
            return label, layer
        else:
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
                num_vx_seg = len(dm_in_seg.ravel())
                num_vx_seg_above_thresh = len(dm_in_seg[dm_in_seg >= gy])
                try:
                    mean_in_seg = np.mean(dm_in_seg[dm_in_seg >= gy]) * act
                    min_in_seg = np.min(dm_in_seg[dm_in_seg >= gy]) * act
                    max_in_seg = np.max(dm_in_seg[dm_in_seg >= gy]) * act
                except Exception as e:
                    mean_in_seg, min_in_seg, max_in_seg = 0, 0, 0

                vol_seg = self.voxel_vol_ml * num_vx_seg_above_thresh
                print(f"\t\tInside {seg_name}:\t\t{vol_seg:.2f} cm3 ({num_vx_seg_above_thresh / num_vx_seg*100:.1f}%) (mean = {mean_in_seg:.2f} Gy), (min / max = {min_in_seg:.1f} / {max_in_seg:.1f} Gy")

        pass


    def plot_XGy_regions(self, X=100, verif=False, inside_segment=None, resamp_to_ct=True, plot_segments={}, crop=(0, 0)):

        # Use meta_spect to interpolate dm to CT-shape?
        # NOTE: sitk cannot handle reading norwegian letters in file-name (ÆØÅ)
        import SimpleITK as sitk
        
        # Only show 100Gy-region inside segmentation inside_segment, should be same as used to calculate total counts / activity
        # TODO: interpolation does not work when inside_segment is not None (?)
        inside_segment = self.segname_tot_counts if inside_segment == None else inside_segment

        if inside_segment:
            seg_total_label, seg_total_layer = self.get_segment_label_and_layer_from_lookup(seg_name=inside_segment, return_layer_as_int=True)
        # SEG_TOTAL = SEG_TOTAL.T         # C-order to F-order (nrrd vs sitk loading)
        # seg_total = sitk.GetImageFromArray(SEG_TOTAL)

        if self.segmentation_path:
            seg_total = sitk.ReadImage(self.segmentation_path)
            SEG_TOTAL = sitk.GetArrayFromImage(seg_total)
            num_layers_seg = SEG_TOTAL.shape[-1]
        # print(SEG_TOTAL.shape, num_layers_seg)

        # print(seg_total.GetSize())
        # print(SEG_TOTAL.shape)
        # print(sitk.GetImageFromArray(SEG_TOTAL).GetSize())
        # print(sitk.GetImageFromArray(SEG_TOTAL[:, :, :, 1]).GetSize())
        # print(seg_total.GetDimension())
        # sys.exit()
        # print(seg_total.shape, np.unique(seg_total, return_counts=True))
        # print(self.CT_path)
        ct = sitk.ReadImage(self.CT_path)

        if verif:
            spect_pet = sitk.ReadImage(self.PET_path)
            print(f"\nPLOTTING {X}Gy-regions for {self.adm_act} GBq administerred 90Y", end="\t")
            self.act_levels = np.array([1])   # !! Important to keep X Gy limit at X, invariant to X / self.act_levels as voxel units are already Gy (NOT per GBq as for work-up)
            spect_pet_name = "PET"
        else:
            print(f"\nPLOTTING {X}Gy-regions for {self.act_levels} GBq administerred 90Y", end="\t")
            spect_pet = sitk.ReadImage(self.SPECT_path)
            spect_pet_name = "SPECT"
        print(f"(inside {inside_segment})" if inside_segment else "")

        dm = sitk.ReadImage(self.dosemap_path)
        if self.segmentation_path:
            print("\tseg, dm, spect/pet, ct (original):\t", sitk.GetArrayFromImage(seg_total).shape, sitk.GetArrayFromImage(dm).shape, sitk.GetArrayFromImage(spect_pet).shape, sitk.GetArrayFromImage(ct).shape)
        else:
            print("\tdm, spect/pet, ct (original):\t", sitk.GetArrayFromImage(dm).shape, sitk.GetArrayFromImage(spect_pet).shape, sitk.GetArrayFromImage(ct).shape)


        ct_size, ct_spacing = ct.GetSize(), ct.GetSpacing()
        # print(ct_size, ct_spacing)
        ct_vol = np.prod(ct_spacing) * 1e-3 # mm3 to cm3
        spect_pet_vol = np.prod(spect_pet.GetSpacing()) * 1e-3
        print(f"\tRESAMPLING DOSEMAP from CT to {spect_pet_name} with voxel volume ratio = {ct_vol / spect_pet_vol:.3g} ({ct_vol:.2e} / {spect_pet_vol:.2e})")

        if resamp_to_ct:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ct)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetSize(ct_size)
            resampler.SetOutputSpacing(ct_spacing)
            resampler.SetTransform(sitk.AffineTransform(3))

            dm_resamp = resampler.Execute(dm)

            # seg_total_resamp = resampler.Execute(seg_total)
            if self.segmentation_path:
                seg_total_resamp = [resampler.Execute(sitk.GetImageFromArray(SEG_TOTAL[:, :, :, d])) for d in range(num_layers_seg)]
            # print(SEG_TOTAL.shape)
            # print([np.count_nonzero(L) / np.prod(L.shape) for L in SEG_TOTAL.T])


            CT = sitk.GetArrayFromImage(ct)
            SPECT = sitk.GetArrayFromImage(spect_pet)
            DM_RESAMP = sitk.GetArrayFromImage(dm_resamp)
            # print(CT.shape, DM_RESAMP.shape, np.mean(DM_RESAMP), np.mean(sitk.GetArrayFromImage(dm)))

            # sys.exit()
            # SEG_TOTAL_RESAMP = sitk.GetArrayFromImage(seg_total_resamp)
            print()

            if self.segmentation_path:
                SEG_TOTAL_RESAMP = np.array([sitk.GetArrayFromImage(seg) for seg in seg_total_resamp])
                # print(SEG_TOTAL_RESAMP.shape, np.count_nonzero(SEG_TOTAL_RESAMP))
                SEG_TOTAL_RESAMP = np.reshape(SEG_TOTAL_RESAMP, (*DM_RESAMP.shape, num_layers_seg))  # THIS IS PROBABLY WRONG
                print(f"\tSEG original: num vx = {np.count_nonzero(SEG_TOTAL)}, \t\tvolume = {np.count_nonzero(SEG_TOTAL) * spect_pet_vol:.1f} mm3")
                print(f"\tSEG resample: num vx = {np.count_nonzero(SEG_TOTAL_RESAMP)}, \tvolume = {np.count_nonzero(SEG_TOTAL_RESAMP) * ct_vol:.1f} mm3")
                print("\tseg, dm, spect/pet, ct (resampeled):\t", SEG_TOTAL_RESAMP.shape, DM_RESAMP.shape, SPECT.shape, CT.shape)
            else:
                print("\tdm, spect/pet, ct (resampeled):\t", DM_RESAMP.shape, SPECT.shape, CT.shape)

        # print(dm.shape)
        # dm_in_tot = dm[seg_total == counts_total_label]
        # print(dm_in_tot.shape)
        if inside_segment:
            print(f"\tVOLUME {X}Gy-region (whole image): {len(DM_RESAMP[DM_RESAMP > X]) * ct_vol:.2f} cm3")
            DM_RESAMP[SEG_TOTAL_RESAMP[:, :, :, seg_total_layer] != seg_total_label] = 0
            print(f"\tVOLUME {X}Gy-region (inside {inside_segment}): {len(DM_RESAMP[DM_RESAMP > X]) *ct_vol:.2f} cm3")


        # print(self.ind_window)  # on SPECT
        ind_window_ct = np.array(self.ind_window) * CT.shape[0] / SPECT.shape[0]
        ind_window_ct = ind_window_ct
        indices = np.linspace(ind_window_ct[0], ind_window_ct[1], 9).astype(int)


        # print(self.act_levels)
        # x_crop = 50
        # y_crop = 10
        x_crop, y_crop = crop
        # xmin, xmax = x_crop, CT.shape[1] - x_crop
        # ymin, ymax = y_crop, CT.shape[1] - y_crop
        xmin, xmax = x_crop
        ymin, ymax = y_crop
        # print(xmin, xmax, ymin, ymax)
        # sys.exit()

        # ind = 35
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15.75, 10))
        ax = ax.ravel()
        # print(CT.shape, DM_RESAMP.shape, indices)

        for i, ind in enumerate(indices):
            # ax[i].imshow(CT[ind, :, :], alpha=1, vmin=-135, vmax=215, cmap="gray")
            ax[i].imshow(CT[ind, xmin:xmax, ymin:ymax], alpha=1, vmin=-135, vmax=215, cmap="gray")
            # plt.imshow(SPECT_RESAMP[ind, :, :], alpha=0.50, cmap="hot")
            # dm_ma = np.ma.masked_where(np.logical_not(DM_RESAMP[ind, :, :]), DM_RESAMP[ind, :, :])

            # plt.imshow(dm_ma, alpha=0.50, cmap="hot")
            # gy_thresholds = np.array([X, X, X]) / self.act_levels
            gy_thresholds = np.repeat(X, len(self.act_levels)) / self.act_levels
            # levels = np.sort(levels)
            colors = ["cyan", "yellow", "red"]
            # print(self.act_levels)
            # print(gy_thresholds)

            # dm_slice = DM_RESAMP[ind, :, :]
            dm_slice = DM_RESAMP[ind, xmin:xmax, ymin:ymax]
            # plt.imshow(dm_slice, alpha=0.50, cmap="hot")
            ax[i].contour(dm_slice, levels=gy_thresholds, colors=colors)

            # ax[i].contour(SEG_TOTAL_RESAMP[ind, :, :, seg_total_layer], levels=[0.5], colors="green")


            c = 0
            for seg in plot_segments.keys():
                seg_label, seg_layer = self.get_segment_label_and_layer_from_lookup(seg_name=seg, return_layer_as_int=True)
                # SEG = SEG_TOTAL_RESAMP[ind, :, :, seg_layer].copy()
                SEG = SEG_TOTAL_RESAMP[ind, xmin:xmax, ymin:ymax, seg_layer].copy()

                # if i == 4:
                #     print(seg, seg_label, seg_layer)
                #     print(np.unique(SEG), seg_label)


                SEG[SEG != seg_label] = 0
                ax[i].contour(SEG, levels=[seg_label-.01], colors=[plot_segments[seg]])
                # ax[i].contour(SEG, levels=[seg_label], colors=[plot_segments[seg]])
                c += 1

                # if i == 4:
                #     print(np.unique(SEG), seg_label)


            ax[i].axis("off")

            # if i == 3:
            #     plt.close()
            #     fig, ax = plt.subplots(figsize=(12,12))
            #     ax.axis("off")
            #     ax = [None, None, None, None, ax]
            # if i == 4:
            #     fig.tight_layout()
            #     plt.show()

        # fig.tight_layout()
        pad = -.1
        fig.subplots_adjust(wspace=pad, hspace=pad)
        # print(" ".join(self.act_levels))
        fig.suptitle(f"{X}Gy regions for {''', '''.join(self.act_levels.astype(str))} GBq administerred 90Y")
        plt.show()


        pass


    def make_dosemap_decay_corrected(self, reference_geometry = "PET", save_dosemap=True):

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

        adm_act_lsfcorr = self.adm_act * (1 - self.shunt_factor)

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
        # print(self.voxel_vol_ml)
        # print((d/10)**3)
        # sys.exit()
        activity_map *= 1e-9    # Bq to GBq

        if self.segname_tot_counts == None:
            sum_act = np.sum(activity_map)
            print(f"\tSum of activity in whole image (segname_tot_count={self.segname_tot_counts} --> no LSF correction) = "
                  f"{sum_act:.3e} Bq ({sum_act / (self.adm_act)*100:.1f} % of {self.adm_act} GBq administerred)")

        else:
            counts_total_index, counts_total_layer = self.get_segment_label_and_layer_from_lookup(self.segname_tot_counts)
            print("\t", counts_total_layer.shape, input_data.seg.shape)
            # sum_act = np.sum(activity_map[counts_total_layer == counts_total_index]) / (1-self.shunt_factor)
            sum_act = np.sum(activity_map[counts_total_layer == counts_total_index])
            # print(f"\tSum of activity in {self.segname_tot_counts} ({self.shunt_factor*100:.0f}% LSF-corrected) = "
            #       f"{sum_act:.3e} Bq ({sum_act / (self.adm_act)*100:.1f} % of {self.adm_act} GBq administerred)")
            print(f"\tSum of activity in {self.segname_tot_counts}:"
                  f" = {sum_act:.2e} Bq -> {sum_act / (adm_act_lsfcorr)*100:.1f} % of {adm_act_lsfcorr:.2f} GBq administerred"
                  f" ({self.shunt_factor*100:.0f}% LSF-corrected from {self.adm_act} GBq)")


        dose_map = activity_map * np.log(2)**-1 * sirt.HALF_LIFE_SEC * sirt.S_FACTOR * s_scaling_factor

        for seg_nm in self.seg_lookup.index.values:
            # counts_total_index, counts_total_layer = get_segment_label_and_layer_from_lookup(input_data.seg, seg_nm, input_data.seg_lookup)
            counts_seg_index, counts_seg_layer = self.get_segment_label_and_layer_from_lookup(seg_nm)
            seg_data = input_array[counts_seg_layer == counts_seg_index]
            # print(seg_data.shape, counts_seg_layer.shape)

            # counts_seg = np.sum(seg_data)
            num_vox_seg = len(seg_data)
            vol_seg = input_data.voxel_vol_ml * num_vox_seg

            # act_seg = np.sum(activity_map[counts_seg_layer == counts_seg_index]) / (1-self.shunt_factor)
            act_seg = np.sum(activity_map[counts_seg_layer == counts_seg_index])

            dose_seg = dose_map[counts_seg_layer == counts_seg_index]

            print(f"\t{seg_nm} ({vol_seg:.2f} mL):\t\ttotal activity = {act_seg*1e3:.2e} MBq ({act_seg / adm_act_lsfcorr * 100:.1f}% of {adm_act_lsfcorr:.2f} GBq LSF-corr adm)"
                  f"\tmean / median dose {np.mean(dose_seg):.2f} / {np.median(dose_seg):.2f} Gy")
        if save_dosemap:
            nrrd.write(input_data.dosemap_path, dose_map, header=input_header, index_order='C')
            print("\tDOSEMAP saved as:", input_data.dosemap_path)
        return dose_map


    def plot_volume_covered_by_x_gy_for_activity(self, dm, X=100, segment_names=None, act_min=0.01, act_max=10):
        print(f"\nCalculating volume covered by {X}Gy given various activity 90Y administerred")

        if type(X) == int:
            fig, ax = plt.subplots()
        elif type(X) == dict:
            fig, axes = plt.subplots(ncols=len(set(X.values())))
            # print(axes)
            # print(np.unique(X.values()))
            # print(len(axes))
            # sys.exit()
            ax_dict = {}
            for i, X_thresh in enumerate(set(X.values())):
                ax_dict[X_thresh] = axes[i]

        else:
            print("NO")
            sys.exit()

        # segment_names = self.seg_lookup.index.values
        if segment_names == None and type(X) == int:
            segment_names = self.seg_operations.index.values
        elif segment_names == None and type(X) == dict:
            segment_names = X.keys()
        print("\tincluding segments", segment_names)

        # seg_name = self.segname_tot_counts if seg_name == None else seg_name
        for j, seg_name in enumerate(segment_names):
            # print(seg_name)

            if type(X) == int:
                X_seg = X
            else:
                X_seg = X[seg_name]
                ax = ax_dict[X_seg]

            # X_seg = X if X == int else X[seg_name]
            act_levels = np.linspace(act_min, act_max, 100)
            # print(act_levels)

            seg_label, seg_layer = self.get_segment_label_and_layer_from_lookup(seg_name, return_layer_as_int=False)
            dm_in_seg = dm[seg_layer == seg_label]
            vx_in_seg = len(dm_in_seg.ravel())
            vol_seg = vx_in_seg * self.voxel_vol_ml
            ratios_seg = []

            for act in act_levels:
                # gy_thresh = X / act
                gy_thresh = X_seg / act
                # print(act, f"GBq -> thresh = {gy_thresh:.1f} Gy / GBq @ {X} Gy", end="\t")
                vx_above_thresh = len(dm_in_seg[dm_in_seg >= gy_thresh])
                ratio = vx_above_thresh / vx_in_seg
                # print(f"{ratio*100:.1f}% above thresh")
                # break
                ratios_seg.append(ratio)


            ax.plot(act_levels, ratios_seg, label=f"{seg_name} ({vol_seg:.0f} mL)", c=f"C{j}")

            ax.set_ylim(0, 1)
            # ax.grid(1)
            ax.set_xlabel("Administered activity $^{90}$Y [GBq]")
            ax.set_ylabel(f"Volume fraction $\geq$ {X_seg} Gy")
            ax.legend()
            # ax.set_title(f"Volume fractions for segmentations above {X}Gy")
        plt.show()

        return 1

    def make_dvh(self, dm, segment_names=None, save=False):
        if segment_names == None:
            segment_names = self.seg_operations.index.values
        print(f"\nMaking cDVH for segmentations:", segment_names)

        fig, ax = plt.subplots()

        dose_vals = np.arange(0, np.max(dm), 1)
        df_dvh = pd.DataFrame(columns=segment_names, index=dose_vals, dtype=float)

        for seg_name in segment_names:
            seg_label, seg_layer = self.get_segment_label_and_layer_from_lookup(seg_name)
            dm_in_seg = dm[seg_layer == seg_label]
            vx_in_seg = len(dm_in_seg.ravel())

            fractions = []
            for d in dose_vals:
                vx_above = dm_in_seg[dm_in_seg >= d]
                frac = len(vx_above) / vx_in_seg
                fractions.append(frac)

            ax.plot(dose_vals, fractions, label=seg_name)

            df_dvh.loc[:, seg_name] = fractions

        ax.legend()
        ax.set_title("Cumulative dose-volume histogram")
        ax.set_xlabel("Gy / GBq")
        ax.set_ylabel("Volume fraction")

        if save:
            dvh_path = os.path.join(self.patient_top_dir, "dvh.csv")
            df_dvh.to_csv(dvh_path)
            print("\tDVH saved as:", dvh_path)
        plt.show()

        pass



def make_dosemap(input_data: InputDataSIRT, shunt_factor: float = 0.0, reference_geometry = "SPECT"):

    print("\nCREATING dosemap using fraction method with", reference_geometry, end="\t")

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



    total_counts = total_counts / (1 - shunt_factor)

    print("\tTOTAL COUNTS:", total_counts, f" = {total_counts / np.sum(input_array)*100:.1f}% of counts in whole image, in {num_vox_in_total_counts / num_vox_in_whole_image * 100:.1f} % of voxels ({vol_total_counts:.2f} mL)")


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
    print("\tdose_factor=", dose_factor)
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

        print(f"\t{seg_nm} ({vol_seg:.2f} mL): {seg_total/total_counts * 100:.1f}% of used counts, with mean dose {np.mean(dose_seg):.2f} Gy / GBq, min / max = {np.min(dose_seg):.1f} / {np.max(dose_seg):.1f} Gy / GBq")

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


def compare_dvh_workup_verif(patient_top_dir:str, act_scale_workup=1.0, include_seg=None):
    path_dvh_workup = os.path.join(patient_top_dir, "workup", "dvh.csv")
    path_dvh_verif = os.path.join(patient_top_dir, "verif", "dvh.csv")



    df_workup = pd.read_csv(path_dvh_workup, index_col=0)   # Gy / GBq
    df_verif = pd.read_csv(path_dvh_verif, index_col=0)
    
    print("COMPARING cDVH work-up to verif:")
    # Scale workup from Gy / GBq to Gy using administerred activity in therapy (GBq)
    print("\twork-up:", df_workup.shape, df_workup.columns.values)
    print("\tverif:", df_verif.shape, df_verif.columns.values)
    print(f"\tscaling work-up to {act_scale_workup} GBq administered 90Y")
    df_workup.index = df_workup.index * act_scale_workup


    dose_vals = df_workup.index.values if df_workup.index.max() < df_verif.index.max() else df_verif.index.values

    # print(df_verif.index[df_verif.index <= max(dose_vals)])
    df_workup = df_workup.loc[df_workup.index <= max(dose_vals)]
    df_verif = df_verif.loc[df_verif.index <= max(dose_vals)]
    print("COMPARING cDVHs from workup to post-therapy")
    print(df_verif.shape, df_workup.shape)

    fig, ax = plt.subplots()

    if include_seg == None:
        include_seg = df_workup.columns

    for i, seg in enumerate(include_seg):
        c = f"C{i}"
        # ax.plot(df_workup.index, df_workup[seg], ":", label=f"{seg} (workup)", c=c)
        # ax.plot(df_verif.index, df_verif[seg], label=f"{seg} (verif)", c=c)

        # Label == name
        # ax.plot(df_workup.index, df_workup[seg], ":", c=c)
        # ax.plot(df_verif.index, df_verif[seg], label=f"{seg}", c=c)

        # Label == verif / workup
        ax.plot(df_workup.index, df_workup[seg], ":", c=c, label="Work-up")
        ax.plot(df_verif.index, df_verif[seg], label=f"Post-therapy", c=c)

        # print(df_workup[seg])
        print(f"MEDIAN workup =", np.median(df_workup[seg]), f"verif =", np.median(df_verif[seg]))


    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume fraction")
    ax.legend()

    plt.show()

    pass


logger = sirt.logger

logger.info("Testing functions")

# seg_name = "Segmentation.seg.nrrd"
seg_name = None
segname_tot_counts = None
settings_name = None

# patient_top_dir = r"C:\Users\toral\OneDrive\OUS\SIRT_dev\BS50\work-up"
# patient_top_dir = r"E:\SIRT\EKR52\Slicer"
# patient_top_dir = r"E:\SIRT\AAMSW82\Slicer"
# patient_top_dir = r"E:\SIRT\AAMSW82\verif"; segname_tot_counts="Liver"; seg_name="dosemap_segmentation.seg.nrrd"
# patient_top_dir = r"C:\Users\toral\OneDrive\OUS\SIRT\EKR52\verif"; segname_tot_counts=None
# patient_top_dir = r"C:\Users\toral\OneDrive\OUS\SIRT\EKR52\Slicer"; segname_tot_counts=None; seg_name="SegmentLabel_johan.seg.nrrd"
# patient_top_dir = r"E:\SIRT\OBS42\verif"; segname_tot_counts="Liver"; seg_name="Segmentation.seg.nrrd"
# patient_top_dir = r"C:\Users\toral\OneDrive\OUS\SIRT\OBS42\workup"; segname_tot_counts="Liver"; seg_name="Segmentation.seg.nrrd"
# patient_top_dir = r"C:\Users\toral\OneDrive\OUS\SIRT\OBS42\verif"; segname_tot_counts="Liver"; seg_name="Segmentation.seg.nrrd"
patient_top_dir = r"C:\Users\toral\OneDrive\OUS\SIRT\PCO59\verif"


input_data = InputDataSIRT(patient_top_dir=patient_top_dir,
                           # segname_tot_counts="Liver",
                           segname_tot_counts=segname_tot_counts,
                           # segname_tot_counts="TotalCounts",
                           segmentation_name=seg_name,
                           # segmentation_name="Segmentation.seg.nrrd",
                           settings_name=settings_name)

verif = True
save_dm = True
save_dvh = True
# input_data.check_files()  # TODO: do this somewhere

# alias_dict = input_data.segmentation.tumor_alias_dict

# MANUAL entries if no setting.csv:
input_data.time_to_img = 2.25
input_data.shunt_factor = 0.05
input_data.ind_window = [0, 62]
input_data.adm_act = 3.053
print(f"\tMANUAL entries: time_to_img = {input_data.time_to_img} hrs, shunt_factor = {input_data.shunt_factor}, adm_act = {input_data.adm_act} GBq")


if not verif:
    dose_map = make_dosemap(input_data, shunt_factor=input_data.shunt_factor, reference_geometry="SPECT")
else:
    dose_map = input_data.make_dosemap_decay_corrected(save_dosemap=save_dm)

# dose_map, _ = nrrd.read(os.path.join(patient_top_dir, "dosemap_LSF-0.04_sum-LIVER_redistr.nrrd"), index_order="C")
# print(dose_map.shape)


from sirt_functions import dose_map_func
# dose_map_func(input_data.SPECT_path, output_path="dose_map_old.nrrd", shunt_factor=0.02)

# sys.exit()

if verif:
    input_data.act_levels = np.array([1.0])
else:
    input_data.act_levels = np.array([5, 4, 3])

# input_data.act_levels = np.array([4, 3, 2])

# input_data.calculate_XGy_region_volumes(dose_map, X=100)
# input_data.calculate_XGy_region_volumes(dose_map, X=40)

# input_data.plot_volume_covered_by_x_gy_for_activity(dose_map, segment_names=["TumorLobe"], X=100)
# input_data.plot_volume_covered_by_x_gy_for_activity(dose_map, X={"Tumour":100, "Left liver":40, "Right liver without tumour":40})
# input_data.plot_volume_covered_by_x_gy_for_activity(dose_map, X={"Tumour":100})
# make_cDVH(dose_map)
# input_data.make_dvh(dose_map, segment_names=["TumorLobe"], save=save_dvh)

# compare_dvh_workup_verif(patient_top_dir=os.path.join(patient_top_dir, ".."),
#                          act_scale_workup=input_data.adm_act, include_seg=["Tumour"])

# input_data.ind_window = [30, 85]
# input_data.plot_XGy_regions(verif=verif, plot_segments={"LeftLobe":"red"})#, "SuperSelective":"green"})
input_data.plot_XGy_regions(verif=verif, inside_segment=False, crop=[[100, 512-100], [50, 512-25]])
# input_data.plot_XGy_regions(verif=verif, inside_segment=False, crop=[[100, 512-100], [50, 512-25]])
sys.exit()
# input_data.plot_XGy_regions(verif=verif, plot_segments={"Liver":"green", "Tumor region":"red"})


# calculate_bq_for_segmentations_operations(dose_map, input_data)
# input_data.seg_operations


segment_voxels = voxel_values_by_segment_name(input_data, dose_map, "Tum1")
make_cDVH(segment_voxels, )

# print(alias_dict)

# sys.exit()


#input_data.check_files()

sys.exit()
