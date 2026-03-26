"""Convert Seaglider basestation files to OG1 format.

This module provides the core functionality for converting Seaglider basestation
NetCDF files into OG1 (Ocean Gliders 1) format. It handles data processing,
variable renaming, attribute assignments, and dataset standardization.
"""

import logging
import os
from datetime import datetime

import numpy as np
import xarray as xr
from tqdm import tqdm

from seagliderOG1 import readers, tools, utilities, vocabularies, writers

_log = logging.getLogger(__name__)


def convert_to_OG1(list_of_datasets: list[xr.Dataset] | xr.Dataset, contrib_to_append: dict[str, str] | None = None,) -> tuple[xr.Dataset, list[str]]:
    """Convert Seaglider basestation datasets to OG1 format.
    Processes a list of xarray datasets or a single xarray dataset, converts them to OG1 format,
    concatenates the datasets, sorts by time, and applies attributes. Main conversion function that
    processes basestation datasets, applies OG1 standardization, concatenates multiple datasets,
    and adds global attributes.

    Parameters
    ----------
    list_of_datasets : list of xarray.Dataset or xarray.Dataset
        A list of xarray datasets or a single xarray dataset in basestation format.
    contrib_to_append : dict of str, optional
        Dictionary containing additional contributor information to append. Default is None.

    Returns
    -------
    tuple of (xarray.Dataset, list of str)
        A tuple containing:
        - ds_og1 (xarray.Dataset): The concatenated and processed dataset in OG1 format.
        - varlist (list of str): A list of variable names from the input datasets.

    """
    if not isinstance(list_of_datasets, list):
        list_of_datasets = [list_of_datasets]

    processed_datasets = []
    firstrun = True

    varlist = []
    # This would be faster if we concatenated the basestation files first, and then processed them.
    # But we need to process them first to get the dive number, assign GPS (could be after), ?
    for ds1_base in tqdm(list_of_datasets, desc="Processing datasets", unit="dataset"):
        varlist = list(set(varlist + list(ds1_base.variables)))
        ds_new, attr_warnings, sg_cal, dc_other, dc_log = process_dataset(
            ds1_base, firstrun
        )
        if ds_new:
            processed_datasets.append(ds_new)
            firstrun = False
        else:
            _log.warning(
                f"Dataset for dive number {ds1_base.attrs['dive_number']} is empty or invalid."
            )

    ds_og1 = xr.concat(processed_datasets, dim="N_MEASUREMENTS")
    ds_og1 = ds_og1.sortby("TIME")

    # Change format of time into datetime64[ns] to avoid problems with attributes and writing to netcdf
    ds_og1['TIME'] = (ds_og1['TIME'].astype('float64') * 1e9).astype('datetime64[ns]')

    # Apply attributes
    ordered_attributes = update_dataset_attributes(
        list_of_datasets[0], contrib_to_append
    )
    for key, value in ordered_attributes.items():
        ds_og1.attrs[key] = value

    # Construct the platform serial number
    if "platform_id" in ds1_base.attrs:
        PLATFORM_SERIAL_NUMBER = ds1_base.attrs["platform_id"].lower()
    else:
        PLATFORM_SERIAL_NUMBER = "sg000"
    ds_og1["PLATFORM_SERIAL_NUMBER"] = PLATFORM_SERIAL_NUMBER
    ds_og1["PLATFORM_SERIAL_NUMBER"].attrs["long_name"] = "glider serial number"

    # ---- Added some more mandatory variables from OG1 ----
    # Construct the platform model
    PLATFORM_MODEL = "University of Washington Seaglider M1 glider"
    ds_og1["PLATFORM_MODEL"] = PLATFORM_MODEL
    ds_og1["PLATFORM_MODEL"].attrs["long_name"] = "model of the glider"
    ds_og1["PLATFORM_MODEL"].attrs[
        "platform_model_vocabulary"
    ] = "https://vocab.nerc.ac.uk/collection/B76/current/B7600024/"

    # WMO identifier
    if "wmo_identifier" in ds1_base.attrs:
        wmo_id = ds1_base.attrs["wmo_identifier"]
    else:
        wmo_id = "0000000"
    ds_og1["WMO_IDENTIFIER"] = wmo_id
    ds_og1["WMO_IDENTIFIER"].attrs["long_name"] = "wmo id"

    # Trajectory
    ds_og1["TRAJECTORY"] = (
        ds_og1["PLATFORM_SERIAL_NUMBER"] + "_" + ds_og1.attrs["start_date"]
    )
    ds_og1["TRAJECTORY"].attrs["long_name"] = "trajectory name"
    ds_og1["TRAJECTORY"].attrs["cf_role"] = "trajectory_id"

    ds_og1["DEPLOYMENT_LATITUDE"] = xr.DataArray(ds_og1.LATITUDE.values[~np.isnan(ds_og1.LATITUDE)][0],
                                              attrs = {"long_name": "latitude of deployment"})
    ds_og1["DEPLOYMENT_LONGITUDE"] = xr.DataArray(ds_og1.LONGITUDE.values[~np.isnan(ds_og1.LONGITUDE)][0],
                                               attrs = {"long_name": "longitude of deployment"})
    ds_og1["DEPLOYMENT_TIME"] = xr.DataArray(ds_og1.TIME.values[~np.isnan(ds_og1.TIME)][0],
                                             attrs = {"long_name": "time of deployment"})

    # Remove attributes from TIME_GPS
    if "TIME_GPS" in ds_og1.variables:
        ds_og1["TIME_GPS"].attrs = {}
    # ---- -------------------------------------------- ----

    # Update time_coverage attributes
    # EFW note: 2025-01-31
    # CHECK LOGIC HERE: Should we be using the first and last time from the first and last dive?
    # Or is time_coverage_start from the base station file a better time to use?
    # Or is there an earlier TIME_GPS timestamp?
    tstart_in_numpy_datetime64 = ds_og1['TIME'][0]
    tend_in_numpy_datetime64 = ds_og1['TIME'][-1]
    tstart_str = utilities._clean_time_string(
        np.datetime_as_string(tstart_in_numpy_datetime64, unit="s")
    )
    tend_str = utilities._clean_time_string(
        np.datetime_as_string(tend_in_numpy_datetime64, unit="s")
    )
    _log.info("Start of mission from TIME[0]: " + tstart_str)
    _log.info("End of mission from TIME[-1]: " + tend_str)
    ds_og1.attrs["time_coverage_start"] = (
        tstart_str  # ds_og1.TIME[0].values.strftime('%Y%m%dT%H%M%S')
    )
    ds_og1.attrs["time_coverage_end"] = (
        tend_str  # ds_og1.TIME[-1].values.strftime('%Y%m%dT%H%M%S')
    )
    ds_og1.attrs["date_created"] = utilities._clean_time_string(
        ds_og1.attrs["date_created"]
    )

    # Update geospatial attributes
    lat_min = ds_og1.LATITUDE.min().values
    lat_max = ds_og1.LATITUDE.max().values
    lon_min = ds_og1.LONGITUDE.min().values
    lon_max = ds_og1.LONGITUDE.max().values
    ds_og1.attrs["geospatial_lat_min"] = lat_min
    ds_og1.attrs["geospatial_lat_max"] = lat_max
    ds_og1.attrs["geospatial_lon_min"] = lon_min
    ds_og1.attrs["geospatial_lon_max"] = lon_max
    depth_min = ds_og1.DEPTH.min().values
    depth_max = ds_og1.DEPTH.max().values
    ds_og1.attrs["geospatial_vertical_min"] = depth_min
    ds_og1.attrs["geospatial_vertical_max"] = depth_max

    # Construct the unique identifier attribute
    id = f"{PLATFORM_SERIAL_NUMBER}_{ds_og1.start_date}_delayed"
    ds_og1.attrs["id"] = id

    return ds_og1, varlist


def process_dataset(ds1_base: xr.Dataset, firstrun: bool = False) -> tuple[
    xr.Dataset,  # Processed dataset with renamed variables, assigned attributes, and additional information
    list[str],  # List of warnings related to attribute assignments
    xr.Dataset,  # Dataset containing variables starting with 'sg_cal'
    xr.Dataset,  # Dataset containing other variables not categorized under 'sg_cal' or 'dc_log'
    xr.Dataset,  # Dataset containing variables starting with 'log_'
]:
    """Processes a dataset by performing a series of transformations and extractions.

    Parameters
    ----------
    ds1_base : xarray.Dataset
        The input dataset from a basestation file, containing various attributes and variables.
    firstrun : bool, optional
        Indicates whether this is the first run of the processing pipeline. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - ds_new (xarray.Dataset): The processed dataset with renamed variables, assigned attributes,
          converted units, and additional information such as GPS info and dive number.
        - attr_warnings (list[str]): A list of warnings related to attribute assignments.
        - sg_cal (xarray.Dataset): A dataset containing variables starting with 'sg_cal'.
        - dc_other (xarray.Dataset): A dataset containing other variables not categorized under 'sg_cal' or 'dc_log'.
        - dc_log (xarray.Dataset): A dataset containing variables starting with 'log_'.

    Notes
    -----
    - The function performs the following steps:
        1. Handles and splits the inputs:
            - Extracts the dive number from the attributes.
            - Splits the dataset by unique dimensions.
            - Extracts the gps_info from the split dataset.
            - Extracts variables starting with 'sg_cal' (originally from sg_calib_constants.m).
        2. Renames the dataset dimensions, coordinates, and variables according to OG1:
            - Extracts and renames dimensions for 'sg_data_point' (N_MEASUREMENTS).
            - Renames variables according to the OG1 vocabulary.
            - Assigns variable attributes according to OG1 and logs warnings for conflicts.
            - Converts units in the dataset (e.g., cm/s to m/s) where possible.
            - Converts QC flags to int8.
        3. Adds new variables:
            - Adds GPS info as LATITUDE_GPS, LONGITUDE_GPS, and TIME_GPS (increasing the length of N_MEASUREMENTS).
            - Adds the divenum as a variable of length N_MEASUREMENTS.
            - Adds the PROFILE_NUMBER (odd for dives, even for ascents).
            - Adds the PHASE of the dive (1 for ascent, 2 for descent, 3 for between the first two surface points).
            - Adds the DEPTH_Z with positive up.
        4. Returns the processed dataset, attribute warnings, and categorized datasets.

    - The function sorts the dataset by TIME and may exhibit undesired behavior if there are not two surface GPS fixes before a dive.

    """
    # Check if the dataset has 'LONGITUDE' as a coordinate
    ds1_base = utilities._validate_coords(ds1_base)
    if ds1_base is None or len(ds1_base.variables) == 0:
        return xr.Dataset(), [], xr.Dataset(), xr.Dataset(), xr.Dataset()
    # Handle and split the inputs.
    # --------------------------------
    # Extract the dive number from the attributes
    divenum = ds1_base.attrs["dive_number"]
    ### check if the pressure dim and longitude dim are the same
    ### if not, combine them inside the dataset
    longitude_dim = ds1_base["longitude"].dims[0]
    pressure_dim = ds1_base["pressure"].dims[0]
    if pressure_dim != longitude_dim:
        ds1_base = tools.combine_two_dim_of_dataset(
            ds1_base, longitude_dim, pressure_dim
        )
    # Split the dataset by unique dimensions
    split_ds = tools.split_by_unique_dims(ds1_base)

    # Extract the sg_data_point from the split dataset
    ds_sgdatapoint = split_ds[(longitude_dim,)]

    # Extract the gps_info from the split dataset
    ds_gps = split_ds[("gps_info",)]
    # Extract variables starting with 'sg_cal'
    # These will be needed to set attributes for the xarray dataset
    ds_sgcal, ds_log, ds_other = extract_variables(split_ds[()])

    # Repeat the value of dc_other.depth_avg_curr_east to the length of the dataset
    var_keep = ["depth_avg_curr_east", "depth_avg_curr_north", "depth_avg_curr_qc"]
    for var in var_keep:
        if var in ds_other:
            v1 = ds_other[var].values
            vector_v = np.full(len(ds_sgdatapoint["longitude"]), v1)
            ds_sgdatapoint[var] = ([longitude_dim], vector_v, ds_other[var].attrs)

    # Rename variables and attributes to OG1 vocabulary
    # -------------------------------------------------------------------
    # Use variables with dimension 'sg_data_point'
    # Must be after split_ds
    ds_new = standardise_OG10(ds_sgdatapoint, firstrun)

    # Add new variables to the dataset (GPS, DIVE_NUMBER, PROFILE_NUMBER, PHASE)
    # -----------------------------------------------------------------------
    # Add the gps_info to the dataset
    # Must be after split_by_unique_dims and after rename_dimensions
    ds_new = add_gps_info_to_dataset(ds_new, ds_gps)
    # Add the profile number (odd for dives, even for ascents)
    ds_new = tools.assign_profile_number(ds_new, ds1_base)
    # Assign the phase of the dive (must be after adding divenum)
    ds_new = tools.assign_phase(ds_new)
    # Assign DEPTH_Z to the dataset where positive is up.
    ds_new = tools.calc_Z(ds_new)

    # Add sensor information to the dataset - can be done on the concatenated data
    # -----------------------------------------------------------------------------
    ds_sensor = tools.gather_sensor_info(ds_other, ds_sgcal, firstrun)
    ds_new = tools.add_sensor_to_dataset(ds_new, ds_sensor, ds_sgcal, firstrun)

    # To avoid problems, reset the dtype of TIME_GPS
    ds_new['TIME_GPS'] = (ds_new['TIME_GPS'].astype('float64') * 1e9).astype('datetime64[ns]')
    vars_to_remove = vocabularies.vars_to_remove #+ ["TIME_GPS"]
    vars_present_to_remove = [var for var in vars_to_remove if var in ds_new.variables]

    # Drop them
    ds_new = ds_new.drop_vars(vars_present_to_remove)
    if firstrun and vars_present_to_remove:
        _log.warning(f"Variables removed from dataset: {vars_present_to_remove}")
    elif firstrun:
        _log.info("No variables needed to be removed from the dataset.")

    attr_warnings = ""
    return ds_new, attr_warnings, ds_sgcal, ds_other, ds_log

def standardise_OG10(
    ds: xr.Dataset,
    firstrun: bool = False,
    unit_format: dict[str, str] = vocabularies.unit_str_format,
) -> xr.Dataset:
    """Standardize the dataset to OG1 format by renaming dimensions, variables, and assigning attributes.

    Applies OG1 vocabulary for variable names, units, and attributes.
    Performs unit conversions and QC flag standardization.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be standardized.
    firstrun : bool, optional
        Indicates whether this is the first run of the standardization process. Default is False.
    unit_format : dict of str, optional
        A dictionary mapping unit strings to their standardized format.
        Default is vocabularies.unit_str_format.

    Returns
    -------
    xarray.Dataset
        The standardized dataset in OG1 format.

    """
    dsa = xr.Dataset()
    dsa.attrs = ds.attrs
    suffixes = ["", "_qc", "_raw", "_raw_qc"]

    # Set new dimension name
    newdim = vocabularies.dims_rename_dict["sg_data_point"]
    # Make a list with all variables not in the vocabularies, and log a warning for them at the end of the loop
    vars_not_in_vocab = []

    # Rename variables according to the OG1 vocabulary
    for orig_varname in list(ds) + list(ds.coords):
        if "_qc" in orig_varname.lower():
            continue
        if orig_varname in vocabularies.standard_names.keys():
            OG1_name = vocabularies.standard_names[orig_varname]
            var_values = ds[orig_varname].values
            # Reformat units and convert units if necessary
            if "units" in ds[orig_varname].attrs:
                orig_unit = tools.reformat_units_var(ds, orig_varname, unit_format)
                if "units" in vocabularies.vocab_attrs[OG1_name]:
                    new_unit = vocabularies.vocab_attrs[OG1_name].get("units")
                    if orig_unit != new_unit:
                        var_values, _ = tools.convert_units_var(
                            var_values,
                            orig_unit,
                            new_unit,
                            vocabularies.unit1_to_unit2,
                            firstrun,
                        )
            dsa[OG1_name] = ([newdim], var_values, vocabularies.vocab_attrs[OG1_name])
            # Pass attributes that aren't in standard OG1 vocab_attrs
            for key, val in ds[orig_varname].attrs.items():
                if key not in dsa[OG1_name].attrs.keys():
                    dsa[OG1_name].attrs[key] = val

            # Add QC variables if they exist
            for suffix in suffixes:
                variant = orig_varname + suffix
                variant_OG1 = OG1_name + suffix.upper()
                if variant in list(ds):
                    dsa[variant_OG1] = ([newdim], ds[variant].values, ds[variant].attrs)
                    # Should only be the root for *_qc variables
                    if "_qc" in variant:
                        # Convert QC flags to int8 and add attributes
                        dsa = tools.convert_qc_flags(dsa, variant_OG1)
        else:
            dsa[orig_varname] = (
                [newdim],
                ds[orig_varname].values,
                ds[orig_varname].attrs,
            )
            ### Only log a warning for variables that aren't in the vocabularies and aren't in the list of variables to keep or remove
            ### Removed varaiables will be printed in the log as being removed, so no need to log a warning for them here.
            if orig_varname not in (*vocabularies.vars_as_is, *vocabularies.vars_to_remove):
                vars_not_in_vocab.append(orig_varname)

    if firstrun and vars_not_in_vocab:
        _log.warning(
            f"Variables not in OG1 vocabulary and not removed: {vars_not_in_vocab}"
        )
    # Assign coordinates
    dsa = dsa.set_coords(["LONGITUDE", "LATITUDE", "DEPTH", "TIME"])
    dsa = tools.encode_times_og1(dsa)
    dsa = tools.set_best_dtype(dsa)
    return dsa


def extract_variables(ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Split variables from the basestation file that have no dimensions into categorized datasets.

    This function further processes the variables from the basestation file that had no dimensions.
    It categorizes them based on their prefixes or characteristics into three groups:
    variables from `sg_calib_constants`, log files, and other mission/dive-specific values.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset. This function is designed to work on variables from the basestation
        file that had no dimensions, typically after being processed by `split_by_unique_dims`.

    Returns
    -------
    tuple of (xarray.Dataset, xarray.Dataset, xarray.Dataset)
        A tuple containing three xarray Datasets:
        - sg_cal : xarray.Dataset
            Dataset containing variables starting with 'sg_cal_' (originally from `sg_calib_constants.m`).
            The variables are renamed to remove the 'sg_cal_' prefix, so they can be accessed directly
            (e.g., `sg_cal.hd_a`).
        - dc_log : xarray.Dataset
            Dataset containing variables starting with 'log_'. These variables are typically from log files.
        - dc_other : xarray.Dataset
            Dataset containing other mission/dive-specific values. This includes depth-averaged currents
            and other variables like `magnetic_variation`.

    """
    sg_cal_vars = {var: ds[var] for var in ds.variables if var.startswith("sg_cal")}
    divecycle_other = {
        var: ds[var] for var in ds.variables if not var.startswith("sg_cal")
    }
    dc_log_vars = {var: ds[var] for var in divecycle_other if var.startswith("log_")}
    divecycle_other = {
        var: data for var, data in divecycle_other.items() if not var.startswith("log_")
    }

    # Create a new dataset with these variables, renaming to remove the leading 'sg_cal_'
    sg_cal = xr.Dataset(
        {var.replace("sg_cal_", ""): data for var, data in sg_cal_vars.items()}
    )
    dc_other = xr.Dataset(divecycle_other)
    dc_log = xr.Dataset(dc_log_vars)

    return sg_cal, dc_log, dc_other


def add_gps_info_to_dataset(ds: xr.Dataset, gps_ds: xr.Dataset) -> xr.Dataset:
    """Add GPS information (LATITUDE_GPS, LONGITUDE_GPS, TIME_GPS) to the dataset.

    The GPS values will be included within the N_MEASUREMENTS dimension, with non-NaN values
    only when GPS information is available. The dataset will be sorted by TIME.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset with renamed dimensions and variables, representing the main data.
    gps_ds : xarray.Dataset
        The dataset containing GPS information, typically extracted from the original
        basestation dataset.

    Returns
    -------
    xarray.Dataset
        The updated dataset with added GPS information. This includes values for
        LATITUDE_GPS, LONGITUDE_GPS, and TIME_GPS only when GPS information is available.

    Notes
    -----
    - The dataset is sorted by TIME (or ctd_time from the original basestation dataset).
    - If the data are not sorted by time, there may be unintended consequences.
    - The function assumes that the GPS dataset contains variables `log_gps_lon`,
      `log_gps_lat`, and `log_gps_time` for longitude, latitude, and time respectively.
    - The function uses the `sg_data_point` dimension as defined in the OG1 vocabulary.

    """
    # Set new dimension name
    newdim = vocabularies.dims_rename_dict["sg_data_point"]

    # Create a new dataset with GPS information
    gps_ds = xr.Dataset(
        {
            "LONGITUDE": ([newdim], gps_ds["log_gps_lon"].values),
        },
        coords={
            "LATITUDE": ([newdim], gps_ds["log_gps_lat"].values),
            "TIME": ([newdim], gps_ds["log_gps_time"].values),
            "DEPTH": ([newdim], np.full(len(gps_ds["log_gps_lat"]), 0)),
        },
    )
    gps_ds = gps_ds.set_coords("LONGITUDE")

    gps_ds["LATITUDE_GPS"] = (
        [newdim],
        gps_ds.LATITUDE.values,
        vocabularies.vocab_attrs["LATITUDE_GPS"],
        {"dtype": ds["LATITUDE"].dtype},
    )
    gps_ds["LONGITUDE_GPS"] = (
        [newdim],
        gps_ds.LONGITUDE.values,
        vocabularies.vocab_attrs["LONGITUDE_GPS"],
        {"dtype": ds["LONGITUDE"].dtype},
    )
    gps_ds["TIME_GPS"] = (
        [newdim],
        gps_ds.TIME.values,
        vocabularies.vocab_attrs["TIME_GPS"],
        {"dtype": ds["TIME"].dtype},
    )

    # Concatenate ds and gps_ds
    datasets = []
    datasets.append(ds)
    datasets.append(gps_ds)
    ds_new = xr.concat(datasets, dim=newdim, data_vars="all")
    ds_new = ds_new.sortby("TIME")

    return ds_new


##-----------------------------------------------------------------------------------------
## Editing attributes
##-----------------------------------------------------------------------------------------
def update_dataset_attributes(ds: xr.Dataset, contrib_to_append: dict[str, str] | None) -> dict[str, str]:
    """Update the attributes of the dataset based on the provided attribute input.

    Processes contributor information, time attributes, and applies OG1
    global attribute vocabulary in the correct order.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset whose attributes need to be updated.
    contrib_to_append : dict of str or None
        A dictionary containing additional contributor information to append. Default is None.

    Returns
    -------
    dict
        A dictionary of ordered attributes with updated values.

    """
    attr_as_is = vocabularies.global_attrs["attr_as_is"]
    attr_to_add = vocabularies.global_attrs["attr_to_add"]
    attr_to_rename = vocabularies.global_attrs["attr_to_rename"]
    order_of_attr = vocabularies.order_of_attr
    mandatory_attr = vocabularies.global_attrs["attr_mandatory"]

    # Extract creators and contributors and institution, then reformulate strings
    contrib_attrs = get_contributors(ds, contrib_to_append)

    # Extract time attributes and reformat basic time strings
    time_attrs = get_time_attributes(ds)

    # Rename some
    renamed_attrs = extract_attr_to_rename(ds, attr_to_rename)

    # Attributes to keep
    keep_attrs = extract_attr_to_keep(ds, attr_as_is)

    # Combine all attributes
    new_attributes = {
        **attr_to_add,
        **contrib_attrs,
        **time_attrs,
        **renamed_attrs,
        **keep_attrs,
        **attr_to_add,
    }

    # Add mandatory attributes if they are not already present
    for attr in mandatory_attr:
        if attr not in new_attributes:
            new_attributes[attr] = mandatory_attr[attr]

    # Reorder attributes according to vocabularies.order_of_attr
    ordered_attributes = {
        attr: new_attributes[attr] for attr in order_of_attr if attr in new_attributes
    }

    # Add any remaining attributes that were not in the order_of_attr list
    for attr in new_attributes:
        if attr not in ordered_attributes:
            ordered_attributes[attr] = new_attributes[attr]

    return ordered_attributes


def get_contributors(ds: xr.Dataset, values_to_append: dict[str, str] | None = None) -> dict[str, str]:
    """Extract and format contributor information for OG1 attributes.

    Processes creator and contributor information from dataset attributes,
    formats them as comma-separated strings, and handles institution mapping.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing original contributor attributes.
    values_to_append : dict, optional
        Additional contributor information to append.

    Returns
    -------
    dict
        Dictionary with formatted contributor attribute strings.

    """
    # Function to create or append to a list
    def create_or_append_list(existing_list, new_item):
        if new_item not in existing_list:
            new_item = new_item.replace(",", "-")
            existing_list.append(new_item)
        return existing_list

    def list_to_comma_separated_string(lst):
        """Convert a list of strings to a single string with values separated by commas.

        Replace any commas present in list elements with hyphens.

        Parameters
        ----------
        lst : list
            List of strings.

        Returns
        -------
        str
            Comma-separated string with commas in elements replaced by hyphens.

        """
        return ", ".join([item for item in lst])

    new_attributes = ds.attrs

    # Initialize empty lists for creator/contributor information and institutions if they are not present
    names = []
    emails = []
    roles = []
    roles_vocab = []
    insts = []
    inst_roles = []
    inst_vocab = []
    inst_roles_vocab = []
    # Parse the original attributes into lists
    if "creator_name" in new_attributes:
        names = create_or_append_list([], new_attributes["creator_name"])
        emails = create_or_append_list([], new_attributes.get("creator_email", ""))
        roles = create_or_append_list([], new_attributes.get("creator_role", "PI"))
        roles_vocab = create_or_append_list(
            [],
            new_attributes.get(
                "creator_role_vocabulary", "http://vocab.nerc.ac.uk/search_nvs/W08"
            ),
        )
        if "contributor_name" in new_attributes:
            names = create_or_append_list(names, new_attributes["contributor_name"])
            emails = create_or_append_list(
                emails, new_attributes.get("contributor_email", "")
            )
            roles = create_or_append_list(
                roles, new_attributes.get("contributor_role", "PI")
            )
            roles_vocab = create_or_append_list(
                roles_vocab,
                new_attributes.get(
                    "contributor_role_vocabulary",
                    "http://vocab.nerc.ac.uk/search_nvs/W08",
                ),
            )
    elif "contributor_name" in new_attributes:
        names = create_or_append_list([], new_attributes["contributor_name"])
        emails = create_or_append_list([], new_attributes.get("contributor_email", ""))
        roles = create_or_append_list([], new_attributes.get("contributor_role", "PI"))
        roles_vocab = create_or_append_list(
            [],
            new_attributes.get(
                "contributor_role_vocabulary", "http://vocab.nerc.ac.uk/search_nvs/W08"
            ),
        )
    if "contributing_institutions" in new_attributes:
        insts = create_or_append_list(
            [], new_attributes.get("contributing_institutions", "")
        )
        inst_roles = create_or_append_list(
            [], new_attributes.get("contributing_institutions_role", "Operator")
        )
        inst_vocab = create_or_append_list(
            [],
            new_attributes.get(
                "contributing_institutions_vocabulary",
                "https://edmo.seadatanet.org/report/1434",
            ),
        )
        inst_roles_vocab = create_or_append_list(
            [],
            new_attributes.get(
                "contributing_institutions_role_vocabulary",
                "http://vocab.nerc.ac.uk/collection/W08/current/",
            ),
        )
    elif "institution" in new_attributes:
        insts = create_or_append_list([], new_attributes["institution"])
        inst_roles = create_or_append_list(
            [], new_attributes.get("contributing_institutions_role", "PI")
        )
        inst_vocab = create_or_append_list(
            [],
            new_attributes.get(
                "contributing_institutions_vocabulary",
                "https://edmo.seadatanet.org/report/1434",
            ),
        )
        inst_roles_vocab = create_or_append_list(
            [],
            new_attributes.get(
                "contributing_institutions_role_vocabulary",
                "http://vocab.nerc.ac.uk/collection/W08/current/",
            ),
        )

    # Rename specific institution if it matches criteria
    for i, inst in enumerate(insts):
        if all(
            keyword in inst for keyword in ["Oceanography", "University", "Washington"]
        ):
            insts[i] = "University of Washington - School of Oceanography"

    # Pad the lists if they are shorter than names
    max_length = len(names)
    emails += [""] * (max_length - len(emails))
    roles += [""] * (max_length - len(roles))
    roles_vocab += [""] * (max_length - len(roles_vocab))
    insts += [""] * (max_length - len(insts))
    inst_roles += [""] * (max_length - len(inst_roles))
    inst_vocab += [""] * (max_length - len(inst_vocab))
    inst_roles_vocab += [""] * (max_length - len(inst_roles_vocab))

    # Append new values to the lists
    if values_to_append is not None:
        for key, value in values_to_append.items():
            if key == "contributor_name":
                names = create_or_append_list(names, value)
            elif key == "contributor_email":
                emails = create_or_append_list(emails, value)
            elif key == "contributor_role":
                roles = create_or_append_list(roles, value)
            elif key == "contributor_role_vocabulary":
                roles_vocab = create_or_append_list(roles_vocab, value)
            elif key == "contributing_institutions":
                insts = create_or_append_list(insts, value)
            elif key == "contributing_institutions_role":
                inst_roles = create_or_append_list(inst_roles, value)
            elif key == "contributing_institutions_vocabulary":
                inst_vocab = create_or_append_list(inst_vocab, value)
            elif key == "contributing_institutions_role_vocabulary":
                inst_roles_vocab = create_or_append_list(inst_roles_vocab, value)

    # Turn the lists into comma-separated strings
    names_str = list_to_comma_separated_string(names)
    emails_str = list_to_comma_separated_string(emails)
    roles_str = list_to_comma_separated_string(roles)
    roles_vocab_str = list_to_comma_separated_string(roles_vocab)

    insts_str = list_to_comma_separated_string(insts)
    inst_roles_str = list_to_comma_separated_string(inst_roles)
    inst_vocab_str = list_to_comma_separated_string(inst_vocab)
    inst_roles_vocab_str = list_to_comma_separated_string(inst_roles_vocab)

    # Create a dictionary for return
    attributes_dict = {
        "contributor_name": names_str,
        "contributor_email": emails_str,
        "contributor_role": roles_str,
        "contributor_role_vocabulary": roles_vocab_str,
        "contributing_institutions": insts_str,
        "contributing_institutions_role": inst_roles_str,
        "contributing_institutions_vocabulary": inst_vocab_str,
        "contributing_institutions_role_vocabulary": inst_roles_vocab_str,
    }

    return attributes_dict


def get_time_attributes(ds: xr.Dataset) -> dict[str, str]:
    """Extract and clean time-related attributes from the dataset.

    Converts various time formats to OG1-standard YYYYMMDDTHHMMSS format
    and adds date_modified timestamp.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing various attributes.

    Returns
    -------
    dict
        A dictionary containing cleaned time-related attributes.

    """
    time_attrs = {}
    time_attr_list = [
        "time_coverage_start",
        "time_coverage_end",
        "date_created",
        "start_time",
    ]
    for attr in time_attr_list:
        if attr in ds.attrs:
            val1 = ds.attrs[attr]
            if isinstance(val1, (int, float)):
                val1 = datetime.utcfromtimestamp(val1).strftime("%Y%m%dT%H%M%S")
            if isinstance(val1, str) and ("-" in val1 or ":" in val1):
                val1 = utilities._clean_time_string(val1)
            time_attrs[attr] = val1
    time_attrs["date_modified"] = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Handle start_date attribute
    if "start_time" in time_attrs:
        time_attrs["start_date"] = time_attrs.pop("start_time")
    if "start_date" not in time_attrs:
        time_attrs["start_date"] = time_attrs["time_coverage_start"]

    return time_attrs


def extract_attr_to_keep(ds1: xr.Dataset, attr_as_is: list[str] = vocabularies.global_attrs["attr_as_is"]) -> dict[str, str]:
    """Extract attributes to retain unchanged.

    Parameters
    ----------
    ds1 : xarray.Dataset
        Source dataset.
    attr_as_is : list
        Attribute names to retain without modification.

    Returns
    -------
    dict
        Retained attributes.

    """
    retained_attrs = {}

    # Retain attributes based on attr_as_is
    for attr in attr_as_is:
        if attr in ds1.attrs:
            retained_attrs[attr] = ds1.attrs[attr]

    return retained_attrs


def extract_attr_to_rename(
    ds1: xr.Dataset, attr_to_rename: dict[str, str] = vocabularies.global_attrs["attr_to_rename"]
) -> dict[str, str]:
    """Extract and rename attributes according to OG1 vocabulary.

    Parameters
    ----------
    ds1 : xarray.Dataset
        Source dataset.
    attr_to_rename : dict
        Mapping of new_name: old_name for attribute renaming.

    Returns
    -------
    dict
        Renamed attributes.

    """
    renamed_attrs = {}
    # Rename attributes based on values_to_rename
    for new_attr, old_attr in attr_to_rename.items():
        if old_attr in ds1.attrs:
            renamed_attrs[new_attr] = ds1.attrs[old_attr]

    return renamed_attrs


def process_and_save_data(input_location: str, save: bool = False, output_dir: str = ".", run_quietly: bool = True) -> xr.Dataset:
    """Process and save data from the specified input location.

    This function loads and concatenates datasets from the server, converts them to OG1 format,
    and saves the resulting dataset to a NetCDF file. If the file already exists, the function
    will prompt the user to decide whether to overwrite it or not.

    Parameters
    ----------
    input_location : str
        The location of the input data to be processed.
    save : bool, optional
        Whether to save the processed dataset to a file. Default is False.
    output_dir : str, optional
        The directory where the output file will be saved. Default is '.'.
    run_quietly : bool, optional
        If True, suppresses user prompts and assumes 'no' for overwriting files. Default is True.

    Returns
    -------
    xarray.Dataset
        The processed dataset.
    """
    # Load and concatenate all datasets from the server
    ds1_base = readers.load_first_basestation_file(input_location)

    # Convert the list of datasets to OG1
    ds1_og1, varlist = convert_to_OG1(ds1_base)
    output_file = os.path.join(output_dir, ds1_og1.attrs["id"] + ".nc")

    # Check if the file exists and delete it if it does
    if os.path.exists(output_file):
        if run_quietly:
            user_input = "no"
        else:
            user_input = input(
                f"File {output_file} already exists. Do you want to re-run and overwrite it? (yes/no): "
            )

        if user_input.lower() != "yes":
            print(f"File {output_file} already exists. Exiting the process.")
            _log.warning(f"File {output_file} already exists. Exiting the process.")
            ds_all = xr.open_dataset(output_file)
            return ds_all
        elif user_input.lower() == "yes":
            ds_all, varlist = convert_to_OG1(list_datasets)
            os.remove(output_file)
            if save:
                writers.save_dataset(ds_all, output_file)
    else:
        print("Running the directory:", input_location)
        _log.info(f"Running the directory: {input_location}")
        list_datasets = readers.load_basestation_files(input_location)
        ds_all, varlist = convert_to_OG1(list_datasets)
        output_file = os.path.join(output_dir, ds_all.attrs["id"] + ".nc")
        if save:
            writers.save_dataset(ds_all, output_file)

        _log.info("===================================================")
        _log.info("input_var: Variables in original basestation files:")
        for varname in sorted(varlist):
            _log.info(f"{varname}")

        _log.info("=========================================")
        _log.info("output_var: Variables in OG1 format file:")
        for varname in sorted(ds_all.variables):
            _log.info(f"{varname}")

    return ds_all
