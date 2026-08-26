"""Vocabularies and mappings for OG1 format conversion.

This module defines the vocabulary mappings, unit conversions, and attribute
configurations used to convert Seaglider basestation files to OG1 format.
It loads YAML configuration files and defines various dictionaries used
throughout the conversion process.

Key Components:
- Variable name mappings from basestation to OG1 format
- Unit conversion factors and formatting rules
- Global attribute definitions and ordering
- Variables to include/exclude during conversion
- Dimension renaming mappings

Configuration Files:
- OG1_var_names.yaml: Variable name mappings
- OG1_vocab_attrs.yaml: Variable attribute vocabularies
- OG1_sensor_attrs.yaml: Sensor attribute definitions
- OG1_global_attrs.yaml: Global attribute configurations
- OG1_author.yaml: Default author/contributor information

Notes
-----
This module is primarily data-driven and loads most configuration from YAML files
in the config/ directory. The YAML-based approach allows easy modification of
OG1 format requirements without code changes.

"""

import os
import pathlib

import yaml

# Set the directory for yaml files as package directory + 'config/'
script_dir = pathlib.Path(__file__).parent.absolute()
config_dir = os.path.join(script_dir, "config/")

# Dimension renaming: maps basestation dimension names to OG1 standard names
dims_rename_dict = {"sg_data_point": "N_MEASUREMENTS"}

# Preferred units for OG1 format - conversion will be attempted if mapping exists
preferred_units = ["m s-1", "dbar", "S m-1"]

# Unit string standardization: maps various unit representations to preferred format
unit_str_format = {
    "m/s": "m s-1",
    "cm/s": "cm s-1",
    "S/m": "S m-1",
    "mS/cm": "mS cm-1",
    "meters": "m",
    "degrees_Celsius": "Celsius",
    "degreesCelsius": "Celsius",
    "g/m^3": "g m-3",
    "kg/m^3": "kg m-3",
    "g/kg": "g kg-1",
    "seconds": "s",
}

# Unit conversion definitions: each entry defines source unit, target unit, and conversion factor
unit1_to_unit2 = {
    "cm s-1_to_m s-1": {"current_unit": "cm s-1", "new_unit": "m s-1", "factor": 0.01},
    "cm/s_to_m/s": {"current_unit": "cm/s", "new_unit": "m/s", "factor": 0.01},
    "m/s_to_cm/s": {"current_unit": "m/s", "new_unit": "cm/s", "factor": 100},
    "m s-1_to_cm s-1": {"current_unit": "m s-1", "new_unit": "cm s-1", "factor": 100},
    "S/m_to_mS/cm": {"current_unit": "S/m", "new_unit": "mS/cm", "factor": 0.1},
    "S m-1_to_mS cm-1": {"current_unit": "S m-1", "new_unit": "mS cm-1", "factor": 0.1},
    "mS/cm_to_S/m": {"current_unit": "mS/cm", "new_unit": "S/m", "factor": 10},
    "mS cm-1_to_S m-1": {"current_unit": "mS cm-1", "new_unit": "S m-1", "factor": 10},
    "dbar_to_Pa": {"current_unit": "dbar", "new_unit": "Pa", "factor": 10000},
    "Pa_to_dbar": {"current_unit": "Pa", "new_unit": "dbar", "factor": 0.0001},
    "dbar_to_kPa": {"current_unit": "dbar", "new_unit": "kPa", "factor": 10},
    "degreesCelsius_to_Celsius": {
        "current_unit": "degreesCelsius",
        "new_unit": "Celsius",
        "factor": 1,
    },
    "Celsius_to_degreesCelsius": {
        "current_unit": "Celsius",
        "new_unit": "degreesCelsius",
        "factor": 1,
    },
    "m_to_cm": {"current_unit": "m", "new_unit": "cm", "factor": 100},
    "m_to_km": {"current_unit": "m", "new_unit": "km", "factor": 0.001},
    "cm_to_m": {"current_unit": "cm", "new_unit": "m", "factor": 0.01},
    "km_to_m": {"current_unit": "km", "new_unit": "m", "factor": 1000},
    "g/m^3_to_kg/m^3": {"current_unit": "g/m3", "new_unit": "kg/m3", "factor": 0.001},
    "g m-3_to_kg m-3": {"current_unit": "g m-3", "new_unit": "kg m-3", "factor": 0.001},
    "kg/m^3_to_g/m^3": {"current_unit": "kg/m3", "new_unit": "g/m3", "factor": 1000},
    "kg m-3_to_g m-3": {"current_unit": "kg m-3", "new_unit": "g m-3", "factor": 1000},
    "micrograms/liter_to_mg m-3": {"current_unit": "micrograms/liter", "new_unit": "mg m-3", "factor": 1.0},
}

# Variables to exclude from OG1 output (derived variables, duplicates, etc.)
vars_to_remove = [
    "dissolved_oxygen_sat",
    "depth",
    "eng_depth",
    "eng_elaps_t",
    "eng_elaps_t_0000",
    "eng_rec",
    "eng_GC_state",
    "latitude_gsm",
    "longitude_gsm",
    "sound_velocity",
    "eng_sbect_condFreq",
    "eng_sbect_tempFreq",
    "glide_angle_gsm",
    "horz_speed_gsm",
    "north_displacement_gsm",
    "east_displacement_gsm",
    "polar_heading",
    "speed_gsm",
    "vert_speed_gsm",
    "dive_num_cast",
    "density",
    "gsw_sigma3",
    "gsw_sigma4",
    "theta",
    # "time",
]

# Variables to keep unchanged during conversion (currently empty)
vars_as_is = []

# --------------------------------
# Variables + variable attributes
# --------------------------------
# Variable name mappings: basestation variable name -> OG1 standard name
# Based on https://github.com/voto-ocean-knowledge/votoutils/blob/main/votoutils/utilities/vocabularies.py
with open(config_dir + "OG1_var_names.yaml", "r") as file:
    standard_names = yaml.safe_load(file)

# Variable attribute vocabularies for OG1 format
# Reference: http://vocab.nerc.ac.uk/scheme/OG1/current/
with open(config_dir + "OG1_vocab_attrs.yaml", "r") as file:
    vocab_attrs = yaml.safe_load(file)

# Sensor attribute vocabularies for OG1 format
# Reference: http://vocab.nerc.ac.uk/scheme/OG_SENSORS/current/
with open(config_dir + "OG1_sensor_attrs.yaml", "r") as file:
    sensor_vocabs = yaml.safe_load(file)


# --------------------------------
# Global Attributes
# --------------------------------
# Default contributor/author information to append to datasets
with open(config_dir + "OG1_author.yaml", "r") as file:
    contrib_to_append = yaml.safe_load(file)

# Preferred order for global attributes in OG1 files
order_of_attr = [
    "title",  # OceanGliders trajectory file
    "id",  # sg015_20040920T000000_delayed
    "platform",  # sub-surface gliders
    "platform_vocabulary",  # https://vocab.nerc.ac.uk/collection/L06/current/27
    "PLATFORM_SERIAL_NUMBER",  # sg015 --> This should be a variable, not an attribute
    "naming_authority",  # edu.washington.apl
    "institution",  # University of washington
    "internal_mission_identifier",  # p0150003_20040924
    "geospatial_lat_min",  # decimal degree
    "geospatial_lat_max",  # decimal degree
    "geospatial_lon_min",  # decimal degree
    "geospatial_lon_max",  # decimal degree
    "geospatial_vertical_min",  # meter depth
    "geospatial_vertical_max",  # meter depth
    "time_coverage_start",  # YYYYmmddTTHHMMss
    "time_coverage_end",  # YYYYmmddTTHHMMss
    "site",  # MOOSE_T00
    "site_vocabulary",  # to be defined
    "program",  # MOOSE glider program
    "program_vocabulary",  # to be defined
    "project",  # SAMBA
    "network",  # Southern California Coastal Ocean Observing System (SCCOOS)
    "contributor_name",  # Firstname Lastname, Firstname Lastname
    "contributor_role",  # Principal Investigator, Operator
    "contributor_role_vocabulary",  # http://vocab.nerc.ac.uk/collection/W08/current/
    "contributor_email",  # name@name.com, name@name.com
    "contributor_id",  # ORCID, ORCID
    "contributor_role_vocabular",  # http://vocab.nerc.ac.uk/search_nvs/W08/
    "contributing_institutions",  # University of Washington, University of Washington
    "contributing_institutions_vocabulary",  # https://edmo.seadatanet.org/report/544, https://ror.org/012tb2g32
    "contributing_institutions_role",  # PI, Operator
    "contributing_institutions_role_vocabulary",  # https://vocab.nerc.ac.uk/collection/W08/current/
    "uri",  # other universal resource identifiers separated by commas
    "data_url",  # url link to where OG1.0 file is hosted
    "doi",  # data doi for OG1
    "rtqc_method",  # No QC applied
    "rtqc_method_doi",  # n/a
    "web_link",  # url for information rleated to glider mission, multiple urls separated by comma
    "comment",  # miscellaneous information
    "start_date",  # datetime of glider deployment YYYYmmddTHHMMss
    "date_created",  # date of creation of this dataset YYYYmmddTHHMMss
    "featureType",  # trajectory
    "Conventions",  # CF-1.10,OG-1.0
    "date_modified",  # date of last modification of this dataset YYYYmmddTHHMMss
]

# Global attribute configuration for OG1 conversion
# Defines which attributes to keep, rename, or add during conversion
# Compatible with base_station_version 2.8, nodc_template_version_v0.9
with open(config_dir + "OG1_global_attrs.yaml", "r") as file:
    global_attrs = yaml.safe_load(file)
