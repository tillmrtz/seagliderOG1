import logging
import re
from dateutil import parser
from datetime import date

import gsw
import numpy as np
import pandas as pd
import xarray as xr

from seagliderOG1 import vocabularies

_log = logging.getLogger(__name__)


# Variables measured directly by the CTD.
CTD_MEASUREMENT_VARIABLES = {
    "temperature",
    "temperature_raw",
    "conductivity",
    "conductivity_raw",
}

# Variables calculated using CTD measurements.
CTD_CALCULATED_VARIABLES = {
    "theta",
    "salinity",
    "salinity_raw",
    "sound_velocity",
    "sigma_theta",
    "sigma_t",
    "density",
    "density_insitu",
}

# Alternative names used for the same physical instrument.
INSTRUMENT_ALIASES = {
    "sbe41": {"sbe41", "sbect"},
}


def OG1_name_mapping(
    ds: xr.Dataset,
    ds1_base: xr.Dataset,
    ctd_dim: str,
) -> pd.DataFrame:
    """Create a mapping from original variable names to OG1 variable names.

    The function examines the dataset immediately before OG1 standardization and
    creates one table row per variable. Each row contains the original variable
    name, its OG1 name, its associated instrument, the instrument type, and its
    original dimensions.

    Instrument assignment is based on the following precedence:

    1. Variables beginning with ``ctd_`` or using ``ctd_data_point`` are
       assigned to the CTD.
    2. When the CTD uses the generic ``sg_data_point`` dimension, temperature
       and conductivity measurements are assigned to the CTD.
    3. When ``ctd_pressure`` is also available, calculated hydrographic
       variables are assigned to the CTD.
    4. The variable's ``instrument`` attribute is checked.
    5. Dimensions named ``<instrument>_data_point`` are checked.
    6. The variable name is checked for an instrument name or alias.

    Variables beginning with ``ctd_`` are processed first. Consequently, when
    several original variables map to the same OG1 name, the CTD variable keeps
    the base name and subsequent variables receive suffixes such as ``2``,
    ``3``, and ``4``.

    Parameters
    ----------
    ds
        Dataset immediately before calling ``standardise_OG10``.
    ds1_base
        Original basestation dataset. It provides the original dimensions,
        variable attributes, and the global ``instrument`` attribute.
    ctd_dim
        Dimension used by the CTD data, for example ``ctd_data_point`` or
        ``sg_data_point``.

    Returns
    -------
    pandas.DataFrame
        Mapping table with the following columns:
    """
    instruments = ds1_base.attrs.get("instrument", "").split()
    standard_names = vocabularies.standard_names
    sensor_vocabs = vocabularies.sensor_vocabs

    has_ctd_pressure = (
        "ctd_pressure" in ds1_base.variables
        or "ctd_pressure" in ds.variables
    )

    def get_source(variable_name: str) -> xr.DataArray:
        """Get a variable from the original dataset when possible."""
        if variable_name in ds1_base.variables:
            return ds1_base[variable_name]

        return ds[variable_name]

    def get_instrument_names(instrument: str) -> set[str]:
        """Get the lowercase name and aliases for an instrument."""
        return INSTRUMENT_ALIASES.get(
            instrument.lower(),
            {instrument.lower()},
        )

    def get_instrument_type(
        instrument: str | None,
    ) -> str | None:
        """Get an instrument's sensor type from the OG1 vocabulary."""
        if instrument is None:
            return None

        og1_instrument_name = standard_names.get(instrument)
        if og1_instrument_name is None:
            return None

        return sensor_vocabs.get(
            og1_instrument_name,
            {},
        ).get("sensor_type")

    def get_ctd_instrument() -> str | None:
        """Find the instrument identified as the CTD."""
        for instrument in instruments:
            instrument_type = get_instrument_type(instrument)

            if (
                isinstance(instrument_type, str)
                and instrument_type.upper() == "CTD"
            ):
                return instrument

        return None

    ctd_instrument = get_ctd_instrument()

    def is_ctd_associated(
        variable_name: str,
        dimensions: set[str],
    ) -> bool:
        """Determine whether a variable should be assigned to the CTD."""
        lower_name = variable_name.lower()

        # Explicit CTD name or dimension.
        if (
            lower_name.startswith("ctd_")
            or "ctd_data_point" in dimensions
        ):
            return True

        # The additional rules are only needed when the CTD shares the
        # generic sg_data_point dimension with other variables.
        if ctd_dim.lower() != "sg_data_point":
            return False

        if lower_name in CTD_MEASUREMENT_VARIABLES:
            return True

        return (
            has_ctd_pressure
            and lower_name in CTD_CALCULATED_VARIABLES
        )

    def find_instrument(variable_name: str) -> str | None:
        """Find the instrument associated with a variable."""
        source = get_source(variable_name)
        lower_name = variable_name.lower()
        dimensions = {
            dimension.lower() for dimension in source.dims
        }

        if (
            ctd_instrument is not None
            and is_ctd_associated(variable_name, dimensions)
        ):
            return ctd_instrument

        # Prefer an explicit instrument attribute.
        variable_instrument = source.attrs.get("instrument")

        if isinstance(variable_instrument, str):
            lower_attribute = variable_instrument.lower()

            for instrument in instruments:
                if lower_attribute in get_instrument_names(instrument):
                    return instrument

        # Match <instrument>_data_point dimensions.
        for instrument in instruments:
            instrument_names = get_instrument_names(instrument)

            if any(
                f"{name}_data_point" in dimensions
                for name in instrument_names
            ):
                return instrument

        # Match instrument names embedded in the variable name.
        for instrument in instruments:
            instrument_names = get_instrument_names(instrument)

            if any(
                lower_name.startswith(f"{name}_")
                or f"_{name}_" in lower_name
                for name in instrument_names
            ):
                return instrument

        return None

    def get_name_candidates(
        variable_name: str,
        instrument: str | None,
    ) -> list[str]:
        """Generate possible vocabulary names by removing prefixes."""
        prefixes = {
            "eng_",
            "instrument_",
            "ctd_",
        }

        if instrument is not None:
            prefixes.update(
                f"{name}_"
                for name in get_instrument_names(instrument)
            )

        candidates = [variable_name]

        # Iterate over the growing list to support multiple prefixes, such as
        # eng_<instrument>_<variable>.
        for candidate in candidates:
            for prefix in prefixes:
                if candidate.lower().startswith(prefix):
                    stripped_name = candidate[len(prefix):]

                    if (
                        stripped_name
                        and stripped_name not in candidates
                    ):
                        candidates.append(stripped_name)

        return candidates

    def get_og1_base_name(
        variable_name: str,
        instrument: str | None,
    ) -> str | None:
        """Find the first OG1 vocabulary match for a variable."""
        for candidate in get_name_candidates(
            variable_name,
            instrument,
        ):
            og1_name = standard_names.get(candidate)

            if og1_name is not None:
                return og1_name

        return None

    # dict.fromkeys removes possible duplicates while preserving order.
    variable_names = list(
        dict.fromkeys(
            list(ds.data_vars) + list(ds.coords)
        )
    )

    variable_names = [
        name
        for name in variable_names
        if "_qc" not in name.lower()
    ]

    # False sorts before True, placing all ctd_ variables first.
    variable_names.sort(
        key=lambda name: not name.lower().startswith("ctd_")
    )

    mapping = []
    og1_name_counts: dict[str, int] = {}

    for original_name in variable_names:
        source = get_source(original_name)
        instrument = find_instrument(original_name)
        base_og1_name = get_og1_base_name(
            original_name,
            instrument,
        )

        og1_name = None

        if base_og1_name is not None:
            count = og1_name_counts.get(base_og1_name, 0) + 1
            og1_name_counts[base_og1_name] = count

            if count == 1:
                og1_name = base_og1_name
            else:
                og1_name = f"{base_og1_name}{count}"

        mapping.append(
            {
                "original_name": original_name,
                "OG1_name": og1_name,
                "instrument": instrument,
                "instrument_type": get_instrument_type(
                    instrument
                ),
                "original_dimension": ", ".join(source.dims),
            }
        )

    return pd.DataFrame(mapping)


def gather_sensor_info(ds1_base) -> dict:
    """Gathers sensor information from an OG1 base dataset.

    Extracts:
      - sensor names (from global 'instrument' attribute)
      - technical specs (from OG1_sensor_attrs.yaml vocabularies)
      - serial numbers + calibration dates (from calibcomm variables)
      - variables belonging to each sensor

    Parameters
    ----------
    ds1_base : xarray.Dataset
        The raw base dataset containing sensor metadata.
    first_run : bool
        If True, informational print statements are shown.

    Returns
    -------
    dict
        Dictionary with one key per sensor, each containing metadata.

    """
    # -------------------------------------------------------------------------
    # 1. Extract sensor names from the 'instrument' global attribute
    # -------------------------------------------------------------------------
    sensor_dict = {}

    if "instrument" in ds1_base.attrs:
        sensor_names = ds1_base.attrs["instrument"].split()
        # Remove unneeded entries
        if "magnetometer" in sensor_names:
            sensor_names.remove("magnetometer")
        # Initialize dictionary entries
        for sensor in sensor_names:
            sensor_dict[sensor] = {}
    else:

        print(
            "Warning: 'instrument' attribute not found in the dataset. Therefore no sensor information extracted from attributes. "
            "If you have sensor information in the attributes, please add an 'instrument' attribute with the sensor names separated by spaces."
            " For example: ds.attrs['instrument'] = 'sbe41 wlbb2f sbe43'"
        )
        return sensor_dict

    # -------------------------------------------------------------------------
    # 2. Add technical specifications from OG1 vocabularies
    # -------------------------------------------------------------------------
    standard_names = vocabularies.standard_names
    sensor_vocabs = vocabularies.sensor_vocabs

    for sensor in sensor_dict.keys():
        if sensor in standard_names:
            new_name = standard_names[sensor]
            sensor_dict[sensor] = sensor_vocabs[new_name]
            print(
                f"Adding technical specifications for '{new_name}' "
                f"(sensor key: '{sensor}') from OG1_sensor_attrs.yaml"
            )
        else:
            print(
                f"Warning: Sensor '{sensor}' not found in standard names vocabulary. "
                "No technical specifications added."
            )

    # -------------------------------------------------------------------------
    # 3. Extract calibration information (serial number + calibration dates)
    # -------------------------------------------------------------------------
    def get_if_exists(base, varname):
        return base[varname] if varname in base.variables else None

    sensor_nums = len(sensor_dict.keys())
    available_calibcomm = [
        v for v in ds1_base.variables if v.startswith("sg_cal_calibcomm")
    ]
    if len(available_calibcomm) > sensor_nums:
        print(
            f"Warning: More calibration variables found as stated in instrument attributes!\n"
            f"The following calibration variables were found: {available_calibcomm}\n"
            f"But only {sensor_nums} sensors were listed in the instrument attributes: {list(sensor_dict.keys())}"
        )

    for sensor in sensor_dict.keys():

        # --- Determine calibcomm variable name --------------------------------
        calibcomm_str = None
        base = ds1_base
        del_caps = _del_capital_letters(sensor)

        calibcomm_str = None
        var = get_if_exists(base, f"sg_cal_calibcomm_{sensor}")
        if var is not None:
            calibcomm_str = str(var.values.item())

        elif (var := get_if_exists(base, f"sg_cal_calibcomm_{del_caps}")) is not None:
            calibcomm_str = str(var.values.item())

        elif sensor_dict[sensor].get("sensor_type") == "CTD":
            var = get_if_exists(base, "sg_cal_calibcomm")
            calibcomm_str = str(var.values.item()) if var is not None else None

        elif sensor_dict[sensor].get("sensor_type") == "Oxygen":
            var = get_if_exists(base, "sg_cal_calibcomm_optode") or get_if_exists(
                base, "sg_cal_calibcomm_oxygen"
            )
            calibcomm_str = str(var.values.item()) if var is not None else None

        elif sensor_dict[sensor].get("sensor_maker") == "WET Labs":
            var = get_if_exists(base, "sg_cal_calibcomm_wetlabs")
            calibcomm_str = str(var.values.item()) if var is not None else None

        if calibcomm_str is None:
            print(
                f"Warning: No calibration info found for sensor '{sensor}'. "
                f"Available calibration variables: {available_calibcomm}"
            )

        # --- Extract serial number + calibration date --------------------------
        serial_number, cal_info = extract_instrument_info(calibcomm_str)

        sensor_dict[sensor]["sensor_serial_number"] = serial_number
        sensor_dict[sensor]["sensor_calibration_date"] = cal_info

    # -------------------------------------------------------------------------
    # 4. Assign variables to sensors based on naming patterns or dimensions
    # -------------------------------------------------------------------------
    sensor_dict = find_variables_for_sensor(ds1_base, sensor_dict)

    return sensor_dict


def add_sensor_to_dataset(ds_og1, sensor_dict, firstrun=False) -> xr.Dataset:
    """Adds sensor information from the provided sensor dictionary to the OG1 dataset.

    Parameters
    ----------
    ds_og1 : xarray.Dataset
        The OG1 dataset to which sensor information will be added.
    sensor_dict : dict
        A dictionary containing sensor metadata and associated variables.

    Returns
    -------
    xarray.Dataset
        The updated OG1 dataset with new sensor metadata variables added.

    Notes
    -----
    - Each sensor becomes a new dimensionless variable:
          SENSOR_<SENSOR_TYPE>_<SERIAL>
    - Attributes for each sensor variable are added from sensor_dict.
    - `sensor_info["variables"]` is handled later (point 3).

    """
    # -------------------------------------------------------------------------
    # 1. Create dimensionless sensor variables in the dataset
    # -------------------------------------------------------------------------
    for _, sensor_info in sensor_dict.items():

        # Build sensor variable name
        sensor_type = sensor_info["sensor_type"].upper().replace(" ", "_")
        serial = sensor_info["sensor_serial_number"]
        sensor_var_name = f"SENSOR_{sensor_type}_{serial}"

        # Create the variable (dimensionless DataArray)
        if firstrun:
            print(
                f"Adding sensor '{sensor_info['sensor_model']}' to the OG1 dataset with attributes"
            )
        ds_og1[sensor_var_name] = xr.DataArray(sensor_info["sensor_model"])

        # ---------------------------------------------------------------------
        # 2. Add sensor attributes (except 'variables')
        # ---------------------------------------------------------------------
        for attr, value in sensor_info.items():
            if attr == "variables":
                continue
            ds_og1[sensor_var_name].attrs[attr] = value

    # -------------------------------------------------------------------------
    # 3. Assign 'sensor' attribute to sensor-specific variables (later update)
    #    Leave logic untouched for now.
    # -------------------------------------------------------------------------

    return ds_og1


def add_dive_number(ds: xr.Dataset, dive_number: int | None = None) -> xr.Dataset:
    """Add dive number as a variable to the dataset. Assumes present in the basestation attributes.

    Parameters
    ----------
    ds
        The dataset to which the dive number will be added.
    dive_number, optional
        The dive number to add. If None, extracts from dataset attributes.

    Returns
    -------
    xarray.Dataset
        The dataset with the dive number added.

    """
    if dive_number is None:
        dive_number = ds.attrs.get("dive_number", np.nan)

    dive_var = xr.DataArray(
        np.full(ds.sizes["N_MEASUREMENTS"], dive_number),
        dims=["N_MEASUREMENTS"],
        attrs={
            "long_name": "dive number",
            "units": "1",
        },
    )
    return ds.assign(DIVE_NUMBER=dive_var)


def assign_profile_number(ds: xr.Dataset, ds1: xr.Dataset) -> xr.Dataset:
    """Assign profile numbers to measurements based on dive phases (down/up casts).

    This function separates each dive into two profiles: descent (down cast) and
    ascent (up cast) phases. The dive is split at the maximum pressure point,
    with the descent phase getting the dive number and ascent getting dive + 0.5.
    Profile numbers are then calculated as 2 * dive_num_cast - 1.

    Parameters
    ----------
    ds
        The dataset to add profile numbers to.
    ds1
        Dataset containing dive number information in attributes.

    Returns
    -------
    xarray.Dataset
        Dataset with 'dive_num_cast' and 'PROFILE_NUMBER' variables added.

    Notes
    -----
    - Requires pressure variable (PRES, ctd_pressure, Pressure, or pres)
    - Down cast: dive_num_cast = dive number
    - Up cast: dive_num_cast = dive number + 0.5
    - Profile numbers: descent = 2*dive-1, ascent = 2*dive

    """
    # Remove the variable dive_num_cast if it exists
    if "dive_num_cast" in ds.variables:
        ds = ds.drop_vars("dive_num_cast")

    # Initialize the new variable with the same dimensions as dive_num
    ds["dive_num_cast"] = (
        ["N_MEASUREMENTS"],
        np.full(ds.sizes["N_MEASUREMENTS"], np.nan),
    )

    ds = add_dive_number(ds, ds1.attrs["dive_number"])

    # Iterate over each unique dive_num
    for dive in np.unique(ds["DIVE_NUMBER"]):
        # Get the indices for the current dive
        dive_indices = np.where(ds["DIVE_NUMBER"] == dive)[0]
        if len(dive_indices) == 0:
            continue  # Skip if no indices found

        # Find the start and end index for the current dive
        start_index = dive_indices[0]
        end_index = dive_indices[-1]

        # Check for possible pressure variable names in ds, then ds1
        possible_press_names = ["PRES", "ctd_pressure", "Pressure", "pres"]
        press_var = next(
            (var for var in possible_press_names if var in ds.variables), None
        )

        if press_var is None:
            press_var = next(
                (var for var in possible_press_names if var in ds1.variables), None
            )

        if press_var is None:
            raise ValueError(
                "No valid pressure variable (PRES or pressure) found in ds or ds1"
            )

        # Get pressure values from the correct dataset
        pressure_data = ds[press_var] if press_var in ds.variables else ds1[press_var]

        # Find the maximum pressure value between start_index and end_index
        pmax = np.nanmax(pressure_data[start_index : end_index + 1].values)

        # Find the index where PRES attains pmax
        pmax_index = start_index + np.argmax(
            pressure_data[start_index : end_index + 1].values == pmax
        )
        # Assign dive_num to all values up to and including pmax
        ds["dive_num_cast"][start_index : pmax_index + 1] = dive

        # Assign dive_num + 0.5 to values after pmax
        ds["dive_num_cast"][pmax_index + 1 : end_index + 1] = dive + 0.5
        # Remove PROFILE_NUMBER if it exists
        if "PROFILE_NUMBER" in ds.variables:
            ds = ds.drop_vars("PROFILE_NUMBER")
        # Calculate profile number and fill Nan with fill value
        fill_value = -9999
        ds["PROFILE_NUMBER"] = (
            (2 * ds["dive_num_cast"] - 1).fillna(fill_value).astype(int)
        )
        ds["PROFILE_NUMBER"].attrs["_FillValue"] = fill_value
    return ds


def assign_phase(ds: xr.Dataset) -> xr.Dataset:
    """This function adds new variables 'PHASE' and 'PHASE_QC' to the dataset `ds`, which indicate the phase of each measurement. The phase is determined based on the pressure readings ('PRES') for each unique dive number ('dive_num').

    Note: In this formulation, we are only separating into dives and climbs based on when the glider is at the maximum depth. Future work needs to separate out the other phases: https://github.com/OceanGlidersCommunity/OG-format-user-manual/blob/main/vocabularyCollection/phase.md and generate a PHASE_QC.
    Assigns phase values to the dataset based on pressure readings.

    Parameters
    ----------
    ds (xarray.Dataset): The input dataset containing 'dive_num' and 'PRES' variables.

    Returns
    -------
    xarray.Dataset: The dataset with an additional 'PHASE' variable, where:
    xarray.Dataset: The dataset with additional 'PHASE' and 'PHASE_QC' variables, where:
        - 'PHASE' indicates the phase of each measurement:
            - Phase 2 is assigned to measurements up to and including the maximum pressure point.
            - Phase 1 is assigned to measurements after the maximum pressure point.
        - 'PHASE_QC' is an additional variable with no QC applied.

    Note: In this formulation, we are only separating into dives and climbs based on when the glider is at the maximum depth.  Future work needs to separate out the other phases: https://github.com/OceanGlidersCommunity/OG-format-user-manual/blob/main/vocabularyCollection/phase.md and generate a PHASE_QC

    """
    # Determine the correct keystring for divenum
    if "dive_number" in ds.variables:
        divenum_str = "dive_number"
    elif "divenum" in ds.variables:
        divenum_str = "divenum"
    elif "dive_num" in ds.variables:
        divenum_str = "dive_num"
    elif "DIVE_NUMBER" in ds.variables:
        divenum_str = "DIVE_NUMBER"
    else:
        raise ValueError("No valid dive number variable found in the dataset.")
    # Initialize the new variable with the same dimensions as dive_num
    ds["PHASE"] = (["N_MEASUREMENTS"], np.full(ds.sizes["N_MEASUREMENTS"], np.nan))
    # Initialize the new variable PHASE_QC with the same dimensions as dive_num
    ds["PHASE_QC"] = (
        ["N_MEASUREMENTS"],
        np.zeros(ds.sizes["N_MEASUREMENTS"], dtype=int),
    )

    # Iterate over each unique dive_num
    for dive in np.unique(ds[divenum_str]):
        # Get the indices for the current dive
        dive_indices = np.where(ds[divenum_str] == dive)[0]
        # Find the start and end index for the current dive
        start_index = dive_indices[0]
        end_index = dive_indices[-1]

        # Find the index of the maximum pressure between start_index and end_index
        pmax = np.nanmax(ds["PRES"][start_index : end_index + 1].values)

        # Find the index where PRES attains the value pmax between start_index and end_index
        pmax_index = start_index + np.argmax(
            ds["PRES"][start_index : end_index + 1].values == pmax
        )

        # Assign phase 2 to all values up to and including the point where pmax is reached
        ds["PHASE"][start_index : pmax_index + 1] = 2

        # Assign phase 1 to all values after pmax is reached
        ds["PHASE"][pmax_index + 1 : end_index + 1] = 1

        # Assign phase 3 to the time at the beginning of the dive, between the first valid TIME_GPS and the second valid TIME_GPS
        valid_time_gps_indices = np.where(
            ~np.isnan(ds["TIME_GPS"][start_index : end_index + 1].values)
        )[0]
        if len(valid_time_gps_indices) >= 2:
            first_valid_index = start_index + valid_time_gps_indices[0]
            second_valid_index = start_index + valid_time_gps_indices[1]
            ds["PHASE"][first_valid_index : second_valid_index + 1] = 3

    return ds


def _del_capital_letters(string):
    return "".join([char for char in string if not char.isupper()])


def find_variables_for_sensor(ds, sensor_dict):
    """Finds variables in the dataset that belong to each sensor based on naming patterns or dimensions.
    For each sensor, looks for variables that either have an 'instrument' attribute matching the sensor name, or have a dimension named '{sensor_name}_data_point'.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to search for variables.
    sensor_dict : dict
        Dictionary containing sensor metadata, used to identify sensor names.

    Returns
    -------
    dict:
        Updated sensor_dict with a list of variables associated with each sensor.

    """
    for sensor_name in sensor_dict.keys():
        variables = []
        for var_name in ds.variables:
            variable = ds[var_name]
            if (
                "instrument" in variable.attrs
                and variable.attrs["instrument"] == sensor_name
            ):
                ## only keep the part of the variable name that comes after the sensor name, e.g. 'eng_wlbb2fl_sig695nm' becomes 'sig695nm'
                # var_name_clean = var_name.replace("eng_", "").replace(f"{sensor_name}_", "").replace("aander","")
                variables.append(var_name)
            elif f"{sensor_name}_data_point" in variable.sizes:
                # var_name_clean = var_name.replace("eng_", "").replace(f"{sensor_name}_", "").replace("aander","")
                variables.append(var_name)
        sensor_variables = list(set(variables))  # Remove duplicates
        # sensor_variables = [standard_names[var] for var in variables if var in standard_names]
        sensor_dict[sensor_name]["variables"] = sensor_variables

    return sensor_dict


##-----------------------------------------------------------------------------------------------------------
## Calculations for new variables
##-----------------------------------------------------------------------------------------------------------
def calc_Z(ds: xr.Dataset) -> xr.Dataset:
    """Calculate the depth (Z position) of the glider using the gsw library to convert pressure to depth.

    Parameters
    ----------
    ds
        The input dataset containing 'PRES', 'LATITUDE', and 'LONGITUDE' variables.

    Returns
    -------
    xarray.Dataset
        The dataset with an additional 'DEPTH' variable.

    """
    # Ensure the required variables are present
    if "PRES" not in ds.variables or "LATITUDE" not in ds.variables:
        raise ValueError("Dataset must contain 'PRES' and 'LATITUDE' variables.")

    # Convert pressure to depth using gsw (pressure in dbar, latitude in degrees)
    depth = gsw.z_from_p(
        ds["PRES"], ds["LATITUDE"]
    ).compute()  # Compute to handle dask arrays

    # Add depth to dataset
    ds["DEPTH_Z"] = (["N_MEASUREMENTS"], depth.data)
    # Assign the calculated depth to a new variable in the dataset

    ds["DEPTH_Z"].attrs = {
        "units": "meters",
        "positive": "up",
        "standard_name": "depth",
        "comment": "Depth calculated from pressure using gsw library, positive up.",
    }

    return ds


def split_by_unique_dims(ds: xr.Dataset) -> dict:
    """Splits an xarray dataset into multiple datasets based on the unique set of dimensions of the variables.

    Parameters
    ----------
    ds
        The input xarray dataset containing various variables.

    Returns
    -------
    dict
        A dictionary mapping dimension tuples to datasets, each with variables sharing the same set of dimensions.

    """
    # Dictionary to hold datasets with unique dimension sets
    unique_dims_datasets = {}
    # Iterate over the variables in the dataset
    for var_name, var_data in ds.data_vars.items():
        # Get the dimensions of the variable
        dims = tuple(var_data.sizes)

        # If this dimension set is not in the dictionary, create a new dataset
        if dims not in unique_dims_datasets:
            unique_dims_datasets[dims] = xr.Dataset()

        # Add the variable to the corresponding dataset
        unique_dims_datasets[dims][var_name] = var_data

    # Convert the dictionary values to a dictionary of datasets
    return {dims: dataset for dims, dataset in unique_dims_datasets.items()}


def reformat_units_var(
    ds: xr.Dataset, var_name: str, unit_format: dict = vocabularies.unit_str_format
) -> str:
    """Rename units in the dataset based on the provided dictionary for OG1.

    Parameters
    ----------
    ds
        The input dataset containing variables with units to be renamed.
    var_name
        The name of the variable whose units should be reformatted.
    unit_format, optional
        A dictionary mapping old unit strings to new formatted unit strings.

    Returns
    -------
    str
        The reformatted unit string.

    """
    """
    Renames units in the dataset based on the provided dictionary for OG1.

    Parameters
    ----------
    ds (xarray.Dataset): The input dataset containing variables with units to be renamed.
    unit_format (dict): A dictionary mapping old unit strings to new formatted unit strings.

    Returns
    -------
    xarray.Dataset: The dataset with renamed units.
    """
    old_unit = ds[var_name].attrs["units"]
    if old_unit in unit_format:
        new_unit = unit_format[old_unit]
    else:
        new_unit = old_unit
    return new_unit


def reformat_units_str(
    old_unit: str, unit_format: dict = vocabularies.unit_str_format
) -> str:
    """Reformat a unit string based on the provided unit format dictionary.

    Parameters
    ----------
    old_unit
        The original unit string to reformat.
    unit_format, optional
        A dictionary mapping old unit strings to new formatted unit strings.

    Returns
    -------
    str
        The reformatted unit string, or the original if no mapping exists.

    """
    if old_unit in unit_format:
        new_unit = unit_format[old_unit]
    else:
        new_unit = old_unit
    return new_unit


def convert_units_var(
    var_values: np.ndarray,
    current_unit: str,
    new_unit: str,
    unit1_to_unit2: dict = vocabularies.unit1_to_unit2,
    firstrun: bool = False,
) -> tuple[np.ndarray, str]:
    """Convert the units of variables in an xarray Dataset to preferred units.  This is useful, for instance, to convert cm/s to m/s.

    Parameters
    ----------
    ds (xarray.Dataset): The dataset containing variables to convert.
    preferred_units (list): A list of strings representing the preferred units.
    unit1_to_unit2 (dict): A dictionary mapping current units to conversion information.
    Each key is a unit string, and each value is a dictionary with:
        - 'factor': The factor to multiply the variable by to convert it.
        - 'units_name': The new unit name after conversion.

    Returns
    -------
    xarray.Dataset: The dataset with converted units.

    """
    current_unit = reformat_units_str(current_unit)
    new_unit = reformat_units_str(new_unit)

    u1_to_u2 = current_unit + "_to_" + new_unit
    if u1_to_u2 in unit1_to_unit2.keys():
        conversion_factor = unit1_to_unit2[u1_to_u2]["factor"]
        new_values = var_values * conversion_factor
    elif current_unit == new_unit:
        new_values = var_values
    else:
        new_values = var_values
        new_unit = current_unit
        if firstrun:
            _log.warning(
                f"\nNo conversion information found for {current_unit} to {new_unit}"
            )
    #        raise ValueError(f"No conversion information found for {current_unit} to {new_unit}")
    return new_values, new_unit


def convert_qc_flags(dsa: xr.Dataset, qc_name: str) -> xr.Dataset:
    """Convert QC flag variables to proper integer format and update attributes.

    This function converts QC flag variables from string format to int8,
    handles NaN values appropriately, removes 'QC_' prefixes from flag meanings,
    and adds proper metadata including long_name and standard_name.

    Parameters
    ----------
    dsa
        The dataset containing QC flag variables.
    qc_name
        The name of the QC flag variable to process.

    Returns
    -------
    xarray.Dataset
        Dataset with converted QC flag variable and updated attributes.

    Notes
    -----
    Must be called after the main variable has been assigned its OG1 long_name.

    """
    # Must be called *after* var_name has OG1 long_name
    var_name = qc_name[:-3]
    if qc_name in list(dsa):
        # Seaglider default type was a string.  Convert to int8 and take care of NaNs
        # dsa[qc_name].values = dsa[qc_name].values.astype("int8")
        values = dsa[qc_name].values
        # Convert byte strings to regular strings (if necessary)
        if values.dtype.type is np.bytes_:
            values = values.astype(str)

        # Use pandas to handle NaNs safely
        values = pd.to_numeric(
            values, errors="coerce"
        )  # Convert strings to numbers, NaNs stay NaNs
        # Assign back to dataset
        dsa[qc_name].values = values
        ### Set the nan values to 6 (unsampled flag) and convert to int8
        ### Before it had just set all values to 0, which is no change flag
        ### Alternative could be to set to 9 (missing value)
        dsa[qc_name].values = dsa[qc_name].fillna(6).astype("int8")
        # Seaglider default flag_meanings were prefixed with 'QC_'. Remove this prefix.
        if "flag_meaning" in dsa[qc_name].attrs:
            flag_meaning = dsa[qc_name].attrs["flag_meaning"]
            dsa[qc_name].attrs["flag_meaning"] = flag_meaning.replace("QC_", "")
        # Add a long_name attribute to the QC variable
        dsa[qc_name].attrs["long_name"] = (
            dsa[var_name].attrs.get("long_name", "") + " quality flag"
        )
        dsa[qc_name].attrs["standard_name"] = "status_flag"
        # Mention the QC variable in the variable attributes
        dsa[var_name].attrs["ancillary_variables"] = qc_name
    return dsa


def find_best_dtype(var_name: str, da: xr.DataArray) -> type:
    """Determine the optimal data type for a variable based on its name and values.

    Parameters
    ----------
    var_name
        The name of the variable.
    da
        The data array to analyze.

    Returns
    -------
    type
        The recommended numpy data type.

    Notes
    -----
    - Latitude/longitude variables use double precision
    - QC variables use int8
    - Time variables keep original dtype
    - Integer variables are downsized based on value range
    - Float64 variables are converted to float32

    """
    input_dtype = da.dtype.type
    if "latitude" in var_name.lower() or "longitude" in var_name.lower():
        return np.double
    if var_name[-2:].lower() == "qc":
        return np.int8
    if "time" in var_name.lower():
        return input_dtype
    if var_name[-3:] == "raw" or "int" in str(input_dtype):
        if np.nanmax(da.values) < 2**16 / 2:
            return np.int16
        elif np.nanmax(da.values) < 2**32 / 2:
            return np.int32
    if input_dtype == np.float64:
        return np.float32
    return input_dtype


def set_fill_value(new_dtype: type) -> int:
    """Calculate appropriate fill value for integer data types.

    Parameters
    ----------
    new_dtype
        The target integer data type.

    Returns
    -------
    int
        The fill value calculated as 2^(bits-1) - 1.

    """
    fill_val = 2 ** (int(re.findall(r"\d+", str(new_dtype))[0]) - 1) - 1
    return fill_val


def set_best_dtype(ds: xr.Dataset) -> xr.Dataset:
    """Optimize data types across all variables in the dataset to reduce memory usage.

    Parameters
    ----------
    ds
        The dataset to optimize.

    Returns
    -------
    xarray.Dataset
        Dataset with optimized data types and appropriate fill values.

    """
    bytes_in = ds.nbytes
    for var_name in list(ds):
        da = ds[var_name]
        input_dtype = da.dtype.type
        new_dtype = find_best_dtype(var_name, da)
        for att in ["valid_min", "valid_max"]:
            if att in da.attrs.keys():
                da.attrs[att] = np.array(da.attrs[att]).astype(new_dtype)
        if new_dtype == input_dtype:
            continue
        _log.debug(f"{var_name} input dtype {input_dtype} change to {new_dtype}")
        da_new = da.astype(new_dtype)
        ds = ds.drop_vars(var_name)
        if "int" in str(new_dtype):
            fill_val = set_fill_value(new_dtype)
            da_new[np.isnan(da)] = fill_val
            da_new.encoding["_FillValue"] = fill_val
        ds[var_name] = da_new
    bytes_out = ds.nbytes
    _log.debug(
        f"Space saved by dtype downgrade: {int(100 * (bytes_in - bytes_out) / bytes_in)} %",
    )
    return ds


def set_best_dtype_value(value, var_name: str):
    """Determine the best data type for a single value based on its variable name and convert it.

    Parameters
    ----------
    value : any
        The input value to convert.

    Returns
    -------
    converted_value : any
        The value converted to the best data type.

    """
    input_dtype = type(value)
    new_dtype = find_best_dtype(var_name, xr.DataArray(value))

    if new_dtype == input_dtype:
        return value

    converted_value = np.array(value).astype(new_dtype)

    if "int" in str(new_dtype) and np.isnan(value):
        fill_val = set_fill_value(new_dtype)
        converted_value = fill_val

    return converted_value


def encode_times(ds: xr.Dataset) -> xr.Dataset:
    """Encode time variables with standard units and remove problematic attributes.

    Parameters
    ----------
    ds
        Dataset containing time variables to encode.

    Returns
    -------
    xarray.Dataset
        Dataset with properly encoded time variables.

    """
    if "units" in ds.time.attrs.keys():
        ds.time.attrs.pop("units")
    if "calendar" in ds.time.attrs.keys():
        ds.time.attrs.pop("calendar")
    ds["time"].encoding["units"] = "seconds since 1970-01-01T00:00:00Z"
    for var_name in list(ds):
        if "time" in var_name.lower() and not var_name == "time":
            for drop_attr in ["units", "calendar", "dtype"]:
                if drop_attr in ds[var_name].attrs.keys():
                    ds[var_name].attrs.pop(drop_attr)
            ds[var_name].encoding["units"] = "seconds since 1970-01-01T00:00:00Z"
    return ds


def encode_times_og1(ds: xr.Dataset) -> xr.Dataset:
    """Encode time variables according to OG1 format specifications.

    Parameters
    ----------
    ds
        Dataset containing time variables to encode.

    Returns
    -------
    xarray.Dataset
        Dataset with OG1-formatted time variables.

    """
    for var_name in ds.variables:
        if "axis" in ds[var_name].attrs.keys():
            ds[var_name].attrs.pop("axis")
        if "time" in var_name.lower():
            for drop_attr in ["units", "calendar", "dtype"]:
                if drop_attr in ds[var_name].attrs.keys():
                    ds[var_name].attrs.pop(drop_attr)
                if drop_attr in ds[var_name].encoding.keys():
                    ds[var_name].encoding.pop(drop_attr)
            if var_name.lower() == "time":
                ds[var_name].attrs["units"] = "seconds since 1970-01-01T00:00:00Z"
                ds[var_name].attrs["calendar"] = "gregorian"
    return ds


def merge_parts_of_dataset(
    ds: xr.Dataset, dim1: str = "sg_data_point", dim2: str = "ctd_data_point"
) -> xr.Dataset:
    """Merges variables from a dataset along two dimensions, ensuring consistency in coordinates.
    The function first separates the dataset into two datasets based on the specified dimensions,
    renames the second dimension to match the first, and then concatenates them along the first dimension.

    Missing time values are filled with NaN, and the final dataset is sorted by time.


    Parameters
    ----------
    ds: xarray.Dataset
        The input dataset containing both dimensions.
    dim1: str
        Primary dimension name (e.g., 'sg_data_point').
    dim2: str
        Secondary dimension name to be merged into dim1 (e.g., 'ctd_data_point').

    Returns
    -------
    merged_ds: xarray.Dataset
        A merged dataset sorted by time.

    Notes
    -----
    Original author: Till Moritz

    """

    def get_time_var(ds, dim):
        """Finds the appropriate time variable based on dimension naming conventions."""
        prefix = dim.split("_data_point")[0]  # Extract prefix
        time_var = "time" if dim == "sg_data_point" else f"{prefix}_time"
        return time_var if time_var in ds.variables else None

    # Extract variables for each dimension
    vars1 = {var: ds[var] for var in ds.variables if dim1 in ds[var].sizes}
    vars2 = {var: ds[var] for var in ds.variables if dim2 in ds[var].sizes}

    # Create separate datasets
    new_ds1, new_ds2 = xr.Dataset(vars1), xr.Dataset(vars2)

    # Rename time variables to 'time' if present
    time_var1, time_var2 = get_time_var(ds, dim1), get_time_var(ds, dim2)
    if time_var1:
        new_ds1 = new_ds1.rename({time_var1: "time"})
    if time_var2:
        new_ds2 = new_ds2.rename({time_var2: "time"})
    # Ensure "time" is a coordinate
    new_ds1 = new_ds1.set_coords("time") if "time" in new_ds1 else new_ds1
    new_ds2 = new_ds2.set_coords("time") if "time" in new_ds2 else new_ds2

    # Rename dim2 to dim1 for consistency
    new_ds2 = new_ds2.rename({dim2: dim1})

    # Add original dimension as attribute
    for new_ds, original_dim in [(new_ds1, dim1), (new_ds2, dim2)]:
        for var in new_ds.variables:
            new_ds[var].attrs["dimension_info"] = f"Original dimension: {original_dim}"

    # Find max size for primary dimension
    max_size = max(new_ds1.sizes.get(dim1, 0), new_ds2.sizes.get(dim1, 0))

    # Pad function to match sizes along dim1
    def pad_ds(ds, max_size):
        ### the dataset makes problems if variables are integers,
        ### so we convert to float before padding and back to int after padding
        for var in ds.variables:
            if "int" in str(ds[var].dtype):
                ds[var] = ds[var].astype(float)

        pad_size = max_size - ds.sizes.get(dim1, 0)
        if pad_size > 0:
            ds = ds.pad({dim1: (0, pad_size)}, constant_values=np.nan)
        return ds

    new_ds1, new_ds2 = pad_ds(new_ds1, max_size), pad_ds(new_ds2, max_size)

    # Get all unique coordinates across both datasets
    all_coords = set(new_ds1.coords) | set(new_ds2.coords)

    # Ensure both datasets contain the same coordinates, filling missing ones with NaN
    for coord in all_coords:
        if coord not in new_ds1:
            new_shape = (
                (new_ds1.sizes[dim1],)
                if dim1 in new_ds1.sizes
                else (len(new_ds1["time"]),)
            )
            new_ds1[coord] = xr.DataArray(np.full(new_shape, np.nan), dims=dim1)
            new_ds1 = new_ds1.set_coords(coord)
        if coord not in new_ds2:
            new_shape = (
                (new_ds2.sizes[dim1],)
                if dim1 in new_ds2.sizes
                else (len(new_ds2["time"]),)
            )
            new_ds2[coord] = xr.DataArray(np.full(new_shape, np.nan), dims=dim1)
            new_ds2 = new_ds2.set_coords(coord)

    # Concatenate along dim1. Missing values will be filled with NaN.
    merged_ds = xr.concat(
        [new_ds1, new_ds2], dim=dim1, join="inner", combine_attrs="drop_conflicts"
    )

    # Sort by time and drop NaT values
    merged_ds = merged_ds.sortby("time").dropna(dim=dim1, subset=["time"])
    return merged_ds


def merge_datasets_along_time(split_ds, dims_to_merge, first_run=False):
    """Merge a list of xarray Datasets along their time dimension.

    Parameters
    ----------
    split_ds : dict[(str,), xr.Dataset]
        Mapping from (dimension,) to Dataset.

    dims_to_merge : list[str]
        Dimension names to extract and merge.

    Returns
    -------
    xr.Dataset or None
        A time-aligned merged dataset, or None if no datasets were eligible.

    """
    processed_datasets = []

    all_dims = set([dim[0] for dim in split_ds.keys() if len(dim) > 0])
    actually_merged_dims = set()
    for dim in dims_to_merge:

        # ---1. Extract dataset---
        if (dim,) not in split_ds:
            print(f"Skipping {dim}: not found in split_ds.")
            continue

        ds = split_ds[(dim,)].copy()
        old_dim = list(ds.sizes)[0]

        # ---2. Detect datetime64 variable---
        time_vars = [v for v in ds.variables if "datetime64" in str(ds[v].dtype)]
        if not time_vars:
            if first_run:
                print(f"Skipping '{dim}': No datetime64 variable found.")
            continue

        ### if more than one time variable is found, takle ctd_time preferibly, otherwise time or the first one. Delete the other time variables.
        if len(time_vars) > 1:
            if "ctd_time" in time_vars:
                time_var = "ctd_time"
            elif "time" in time_vars:
                time_var = "time"
            else:
                time_var = time_vars[0]
            for var in time_vars:
                if var != time_var:
                    ds = ds.drop_vars(var)
        else:
            time_var = time_vars[0]

        # ---3. Rename detected time variable to 'time'---
        if time_var != "time":
            ds = ds.rename({time_var: "time"})

        # ---4. Swap old dimension to time---
        ds = ds.swap_dims({old_dim: "time"})

        # ---5. Add attribute old_dim to each data variable and coordinate (except the time coordinate)---
        for var in ds.variables:
            if var != "time":
                ds[var].attrs["old_dimension"] = old_dim
        if first_run:
            print(
                f"Adding variables with dimension '{dim}' and time variable '{time_var}'."
            )

        processed_datasets.append(ds)
        actually_merged_dims.add(dim)

    if not processed_datasets:
        print("No datasets processed. Returning None.")
        return None

    # ---6. Merge along shared time coordinate---
    merged_ds = xr.merge(processed_datasets, join="outer")

    # ---7. Swap to N_MEASUREMENTS (optional)---
    merged_ds = merged_ds.swap_dims({"time": "N_MEASUREMENTS"})

    merged_ds = merged_ds.sortby("time")
    if first_run:
        ## Print what remaining dimensions were not merged into the new dataset
        print(
            f"The following dimensions were not merged into the new dataset: {all_dims - actually_merged_dims}"
            "\nIf instrument data is missing make sure it's dimension follows the naming convention of '<instrument>_data_point'"
            "\nfrom the ds.attrs['instrument'] list."
        )

    return merged_ds


def combine_two_dim_of_dataset(
    ds: xr.Dataset, dim1: str = "sg_data_point", dim2: str = "ctd_data_point"
) -> xr.Dataset:
    """Updates the original dataset by removing variables with dim1 and dim2
    and adding the merged dataset.

    Parameters
    ----------
    ds
        The original dataset.
    dim1, optional
        First dimension to be removed. Default is 'sg_data_point'.
    dim2, optional
        Second dimension to be removed. Default is 'ctd_data_point'.

    Returns
    -------
    xarray.Dataset
        The updated dataset with merged variables.

    """
    # Drop all variables that have dim1 or dim2
    vars_to_drop = [
        var for var in ds.variables if dim1 in ds[var].sizes or dim2 in ds[var].sizes
    ]
    cleaned_ds = ds.drop_vars(vars_to_drop, errors="ignore")
    merged_ds = merge_parts_of_dataset(ds, dim1=dim1, dim2=dim2)
    # Merge the cleaned dataset with the merged dataset
    updated_ds = xr.merge([cleaned_ds, merged_ds], combine_attrs="drop_conflicts")

    return updated_ds


standard_names = vocabularies.standard_names


def extract_hdm_parameters(list_datasets):
    """Extracts HDM parameters and their attributes from a list of datasets. If the parameter has the same value across all datasets,
    it keeps a single value; otherwise, it returns the full list.

    Parameters
    ----------
    list_datasets (list): List of xarray.Dataset objects.
    standard_names (dict): Vocabulary mapping internal names to standard names.

    Returns
    -------
    dict: A nested dictionary where keys are standard names and values contain
            the 'data' and 'attributes'.

    """
    potential_parameters_OG1 = [
        "VBD_MIN_CNTS",
        "VBD_CNTS_PER_CC",
        "VBD_CC_PER_CNTS",
        "VBD_BIAS",
        "MASS",
        "VOLMAX",
        "C_VBD",
        "HD_A",
        "HD_B",
        "HD_C",
    ]
    potential_parameters = [
        key
        for key, value in standard_names.items()
        if value in potential_parameters_OG1
    ]
    hdm_variables = {}

    for param in potential_parameters:
        # Determine the key name using standard_names mapping
        param_key = standard_names.get(param, param)
        if param_key == param and param not in standard_names:
            print(
                f"Warning: '{param}' not found in standard names. Using original name."
            )

        # 1. Check if the parameter exists in the datasets
        if param not in list_datasets[0].variables:
            continue

        # 2. Extract values from all datasets
        all_values = [ds[param].values for ds in list_datasets]

        # 3. Determine if we keep a single value or the full list
        # We flatten to handle case where .values might be arrays
        unique_vals = np.unique(np.array(all_values))

        final_value = unique_vals[0] if len(unique_vals) == 1 else all_values

        if isinstance(final_value, list):
            final_value = np.array(final_value)

        # 4. Store as a dictionary to accommodate both value and attributes and add long_name attribute
        hdm_variables[param_key] = {
            "values": final_value,
            "attributes": list_datasets[0][param].attrs,  # Add long_name attribute
        }
        hdm_variables[param_key]["attributes"]["original_name"] = param_key

    # 5. Print what parameters from potential_parameters were found and which couldn't not be found in the datasets
    found_params = [
        param for param in potential_parameters_OG1 if param in hdm_variables
    ]
    not_found_params = [
        param for param in potential_parameters_OG1 if param not in hdm_variables
    ]
    print(f"The following HDM parameters were found: {found_params}")
    if not_found_params:
        print(
            f"Warning: The following potential HDM parameters were not found in the datasets: {not_found_params}"
        )

    # 6. Add dive_number in order to be able to assign dive-based parameters to the correct profiles in the OG1 dataset
    dive_numbers = None
    if "dive_number" in list_datasets[0].attrs:
        dive_numbers = [ds.dive_number.item() for ds in list_datasets]
    elif "trajectory" in list_datasets[0].data_vars:
        dive_numbers = [ds["trajectory"].values for ds in list_datasets]
    if dive_numbers is not None:
        hdm_variables["DIVE_NUMBER"] = {"values": dive_numbers}
    else:
        print(
            "Warning: 'dive_number' or 'trajectory' not found in datasets. "
            "Dive-based parameters may not be correctly assigned."
        )
    return hdm_variables


def add_hdm_parameters(ds_OG1, hdm_parameters):
    """Add HDM parameters to the OG1 dataset as new variables with their attributes.

    Parameters
    ----------
    ds_OG1 (xarray.Dataset): The OG1 dataset to which HDM parameters will be added.
    hdm_parameters (dict): A dictionary containing HDM parameters and their attributes
                            in the format {standard_name: {"value": ..., "attributes": {...}}}.

    Returns
    -------
    xarray.Dataset: Updated OG1 dataset with HDM parameters added as variables.

    """
    ds_updated = ds_OG1.copy()

    dive_numbers = hdm_parameters.pop("DIVE_NUMBER", {}).get("values", None)

    for param_name, param_info in hdm_parameters.items():
        # Using .get() because you used "value" in extract and "values" in your draft
        values = param_info.get("value") or param_info.get("values")
        attributes = param_info["attributes"]

        if values is None or np.size(values) == 0:
            print(f"Warning: Parameter '{param_name}' values are empty. Skipping.")
            continue
        # Check if it's a single value (scalar)
        if dive_numbers is None or np.size(values) == 1:
            val = np.atleast_1d(values)[0]
            ds_updated[param_name] = val.item() if hasattr(val, "item") else val
            ds_updated[param_name].attrs = attributes
        # Check if it's dive-based (1 value per 2 profiles)
        elif np.size(values) > 1:
            mapped_array = np.full(ds_updated.N_MEASUREMENTS.shape, np.nan)

            # Iterate through dives (each dive = 2 profiles)
            for dive, dive_val in zip(dive_numbers, values):
                # Find all measurement indices belonging to these two profiles
                if "DIVE_NUMBER" in ds_updated.data_vars:
                    mask = ds_updated.DIVE_NUMBER == dive
                elif "PROFILE_NUMBER" in ds_updated.data_vars:
                    # Logic: Dive 1 = Profiles 1 & 2
                    mask = (ds_updated.PROFILE_NUMBER == 2 * dive - 1) | (
                        ds_updated.PROFILE_NUMBER == 2 * dive
                    )
                else:
                    print(
                        f"Error: No reference dimension for {param_name}. Skipping dive mapping."
                    )
                    break

                # Fill the array for those specific measurements
                mapped_array[mask] = dive_val

            # Add to dataset with the N_MEASUREMENTS dimension
            ds_updated[param_name] = (("N_MEASUREMENTS",), mapped_array)
            ds_updated[param_name].attrs = attributes

    return ds_updated


def parse_8_digit_date(date_str):
    """Validates and formats 8-digit strings.
    Prioritizes YYYY-MM-DD, then DD-MM-YYYY based on realistic ranges.
    """
    # Extract only the digits
    d = "".join(re.findall(r"\d", date_str))
    if len(d) != 8:
        return None

    # Try YYYYMMDD (Standard ISO-like)
    # Year: 1900-2099, Month: 01-12, Day: 01-31
    y, m, day = int(d[:4]), int(d[4:6]), int(d[6:])
    if 1900 <= y < date.today().year and 1 <= m <= 12 and 1 <= day <= 31:
        return f"{y:04d}-{m:02d}-{day:02d}"

    # Try DDMMYYYY (European style)
    # Day: 01-31, Month: 01-12, Year: 1900-2099
    day, m, y = int(d[:2]), int(d[2:4]), int(d[4:])
    if 1900 <= y < date.today().year and 1 <= m <= 12 and 1 <= day <= 31:
        return f"{y:04d}-{m:02d}-{day:02d}"

    # Try MMDDYYYY (US style)
    m, day, y = int(d[:2]), int(d[2:4]), int(d[4:])
    if 1900 <= y < date.today().year and 1 <= m <= 12 and 1 <= day <= 31:
        return f"{y:04d}-{m:02d}-{day:02d}"

    return "Format Error"


def extract_instrument_info(input_string):

    if (
        input_string is None
        or not isinstance(input_string, str)
        or input_string.strip() == ""
    ):
        return "0000", "00-00-0000"

    s = input_string.replace(",", " ").replace(";", " ")

    # 1. EXTRACT SERIAL NUMBER (DIGITS ONLY)
    sn_patterns = [
        r"(?i)(?:s/n|sn|serial\s*#|serialnum)[:\s]*([\w-]+)",
        r"(?i)SBE\s+([\d-]+)",
        r"(?i)SN([\d-]+)",
    ]

    raw_sn = ""
    for pattern in sn_patterns:
        match = re.search(pattern, s)
        if match:
            raw_sn = match.group(1)
            break

    # Strip letters and dashes, keeping only the numbers
    if raw_sn:
        serial_number = "".join(re.findall(r"\d+", raw_sn))
    else:
        serial_number = "0000"

    # 2. EXTRACT CALIBRATION DATES
    multi_match = re.findall(r"(\w+):(\d{4}-\d{2}-\d{2}T[\d:]+Z)", s)

    if multi_match:
        cal_info = ", ".join(
            [f"{m[0]}: {parser.parse(m[1]).strftime('%Y-%m-%d')}" for m in multi_match]
        )
    else:
        parts = re.split(r"(?i)cal(?:ibration)?[:\s]*", s)
        if len(parts) > 1:
            date_candidate = parts[1].strip()

            # Handle the specific 8-digit date requirement (29082012 -> 2908-20-12)
            digit_only_date = "".join(re.findall(r"\d", date_candidate))
            if len(digit_only_date) == 8 and "?" not in date_candidate:
                cal_info = parse_8_digit_date(date_candidate)
            elif "?" in date_candidate:
                cal_info = "0000-00-00"
            else:
                try:
                    dt = parser.parse(date_candidate, fuzzy=True, dayfirst=True)
                    cal_info = dt.strftime("%Y-%m-%d")
                except:
                    cal_info = "Format Error"
        else:
            cal_info = "None Found"

    ## if cal_info has unrealistic year, month or day, rearrange so that year is first, then try parsing again
    if cal_info not in ["None Found", "Format Error", "0000-00-00"]:
        try:
            dt = parser.parse(cal_info, fuzzy=True, dayfirst=True)
            if (
                dt.year < 1900
                or dt.year > date.today().year
                or dt.month > 12
                or dt.day > 31
            ):
                # Try parsing with year first
                dt = parser.parse(cal_info, fuzzy=True, yearfirst=True)
                cal_info = dt.strftime("%Y-%m-%d")
        except:
            pass

    return serial_number, cal_info


# ===============================================================================
# Unused functions
# ===============================================================================
