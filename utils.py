

def get_layer_label_values_from_meta(meta, segment):
    layer = int(meta[segment + "_Layer"])
    label = int(meta[segment + "_LabelValue"])

    return [layer, label]
