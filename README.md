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
## List of dependencies

The scripts should use fairly standardized python libraries to improve usability. 
As of now the following libraries are required:

- argparse
- numpy 
- nrrd [Get it here](https://pynrrd.readthedocs.io/en/stable/)

## TODOs

See the issues-page for an updated list.

