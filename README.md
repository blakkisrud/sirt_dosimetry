# sirt_dosimetry

Collection of scripts to perform dosimetry based on SPECT and PET-images for selective internal radiation therapy (SIRT).
The scripts and programs are first and foremost for in-house use at Oslo University Hospital, hosting it on github mainly facilitates internal co-programming.

## List of scripts/programs

### create_dose_map

Script to take an nrrd-file with a count-distribution and output a dose map in units of Gy/GBq.
The calculation is based on having the total counts in the image (with or without a correction for lung-shunting) representing the total administered activity.
A local absorbed dose deposition is assumed.
The script works as a command line tool with this example usage:

``
python create_dose_map --input_path input_path --output_path output_path shunt_factor shunt
``
### Class InputDataSIRT

The script may also be run using a class implementation:

```python
sirt = InputDataSIRT(patient_top_dir, segname_tot_counts, segmentation_name, settings_name)
```
Where 
- patient_top_dir is the directory holding all relevant nrrd-files, and a settings.csv file
- if segname_tot_counts != None the script uses only counts from within the (named) segment with subsequent LSF-correction
- segmentation_name: e.g. "Segmentation.seg.nrrd" holding all relevant segmentation, including segmame_tot_counts if used
- settings_name: csv-file with temporal and dose-information, including segmentation names (not necessary), and lung-shunt fraction (LSF), among others (dependencies vary, see code example below)


To calculate the dose-map, two functions may be used - depending on the method:
- For workup-calculations: make_dosemap
	- using LSF and fraction-map method
- For verification: sirt.make_dosemap_decay_corrected
	- decay-correcting the image using time from activity administration to PET-imaging 

```python
if not verif:  
    dose_map = make_dosemap(sirt, shunt_factor=sirt.shunt_factor, reference_geometry="SPECT")  
else:  
    dose_map = sirt.make_dosemap_decay_corrected(reference_geometry="PET", save_dosemap=True)
```


The created dose-map may further be used to:
- Compute volumes receiving more or equal to a given dose (X Gy)
	- may also plot percentage of volume covered by at least X Gy
- Creating cumulative dose-volume histograms (DVH)
	- may be used to compare work-up predictions with post-therapy verification
- Plotting 3x3 grid of within-slice regions receiving more or equal to a given dose (X Gy)

```python
# Calculating volumes recieving 100 Gy or more in whole image
	# if e.g. inside_segment = "Liver" -> calculates volumes inside segmented liver
sirt.calculate_XGy_region_volumes(dose_map, X=100, inside_segment=None)


# Calculate and plot DVH -> save as dvh.csv inside patient_top_dir
	# plots for all loaded segmentations if segment_names = None, 
	# may also specify which to include as list of names
sirt.make_dvh(dose_map, segment_names=None, save=True)


if verif:
	# Compare (saved) DVHs between work-up and post-therapy
	# scales work-up DVH from Gy / GBq to Gy using sirt.adm_act (GBq)
	compare_dvh_workup_verif(patient_top_dir=os.path.join(patient_top_dir, ".."),  
	                         act_scale_workup=sirt.adm_act, include_seg=["Tumour", "Left liver", "Liver not tumor"])
	


# Create plot: volume percantages of segmentations recieving at least X Gy per GBq
	# May vary dose-threshold X between segmentations included
sirt.plot_volume_covered_by_x_gy_for_activity(dose_map, X={"Tumour":100, "Left liver":40, "Liver not tumor":40})


# Create 9x9 plot of within-slice dose-threshold regions recieiving at least X Gy, interpolated from SPECT / PET to CT dimensions using a linear transform
	# required: sirt.ind_window = [a, b] -> determining which (z-dim) slice to start / stop at with equidistanced spacing
	# if verif: units are Gy / GBq, if not verif: Gy based on sirt.act_levels
	# if crop -> reduce FOV for all included 
sirt.plot_XGy_regions(verif=False, X=100, inside_segment=False, crop=[[100, 512-100], [50, 512-25]])

```

## List of dependencies

The scripts should use fairly standardized python libraries to improve usability. 
As of now the following libraries are required:

- argparse
- numpy 
- nrrd [Get it here](https://pynrrd.readthedocs.io/en/stable/)
- yaml

## TODOs

See the issues-page for an updated list.

### Short term

- Script to perform dosimetry on pre-segmented regions

### Longer term

- Decide on graphic user interface
- Embedded segmentation or SLICER-bridging


