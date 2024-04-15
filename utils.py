import numpy as np


def get_layer_label_values_from_meta(meta, segment):
    layer = int(meta[segment + "_Layer"])
    label = int(meta[segment + "_LabelValue"])
    return layer, label

def get_segment_label_and_layer_from_lookup(seg, name, lookup):
    seg_layer = seg[:, :, :, lookup.loc[name, "Layer"]]
    label = lookup.loc[name, "Label"]
    return label, seg_layer


def check_segments_overlaps(seg, seg_tot_name, df_lookup):
    # FOR ALL LABEL NAMES (indices in df_lookup), CHECKS IF TOTALLY CONTAINED IN
    # seg_tot_name USING A SET OF COORDINATES

    seg_layer_tot = seg[:, :, :, df_lookup.loc[seg_tot_name, "Layer"]]
    label_tot = df_lookup.loc[seg_tot_name, "Label"]

    # Create set of coordinates given layer and label for seg_tot_name
    ind_in_tot = set((x, y, z) for x, y, z in np.argwhere(seg_layer_tot == label_tot))

    not_overlap_flag = False

    for seg_nm in df_lookup.index.values:
        seg_layer = seg[:, :, :, df_lookup.loc[seg_nm, "Layer"]]
        label_nm = df_lookup.loc[seg_nm, "Label"]
        ind_in_nm = set((x, y, z) for x, y, z in np.argwhere(seg_layer == label_nm))
        # ind_diff = ind_in_tot.difference(ind_in_nm)
        ind_diff = ind_in_nm.difference(ind_in_tot)
        not_overlap_flag = any(ind_diff)
        # print(seg_nm, not_overlap_flag, len(ind_in_nm), np.count_nonzero(seg_layer[seg_layer == label_nm]))
        if not_overlap_flag:
            print(seg_nm, "NOT CONTAINED IN ", seg_tot_name, "FOR", len(ind_diff), "INDICES...")


    return not(not_overlap_flag)


def calculate_bq_for_segmentations_operations(dm, input):
    import itertools

    seg_oper = input.seg_operations
    print(seg_oper)

    for seg_nm, data in seg_oper.iterrows():
        print(seg_nm, data)

        # break

    pass

