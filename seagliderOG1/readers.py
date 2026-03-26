import os
import pathlib
import re
import sys

import pooch
import requests
import xarray as xr
from bs4 import BeautifulSoup
from tqdm import tqdm

script_dir = pathlib.Path(__file__).parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

"""Readers module for Seaglider basestation files.

This module provides functions to read Seaglider basestation NetCDF files
from either online sources or local directories. It does not manipulate
the data, only loads it into xarray datasets.
"""

# readers.py: Will only read files.  Not manipulate them.

"""# Use pooch for sample files only.
# For the full dataset, just use BeautifulSoup / requests
server = "https://www.ncei.noaa.gov/data/oceans/glider/seaglider/uw/033/20100903/"
data_source_og = pooch.create(
    path=pooch.os_cache("seagliderOG1"),
    base_url=server,
    registry={
        "p0330001_20100903.nc": "1ee726a4890b5567653fd53501da8887d947946ad63a17df4e5efd2e578fb464",
        "p0330002_20100903.nc": "1c0d6f46904063dbb1e74196bc30bdaf6163e7fbd4cc31c6087eb295688a2cc1",
        "p0330003_20100903.nc": "779bdfb4237b17b1de8ccb5d67ef37ea28b595d918b3a034606e8612615058c3",
        "p0330004_20100904.nc": "f981d482e04bbe5085e2f93d686eedf3a30ca125bd94696648d04b2e37fa2489",
        "p0330005_20100904.nc": "46e921c06b407a458dd4f788b9887ea4b6a51376021190c89e0402b25b0c8f3f",
        "p0330006_20100904.nc": "3ed79ee0757b573f32c7d72ff818b82d9810ebbd640d8a9b0f6829ab1da3b972",
        "p0330007_20100905.nc": "29d83c65ef4649bbc9193bb25eab7a137f92da04912eae21d9fc9860e827281d",
        "p0330008_20100905.nc": "66fa25efb8717275e84f326f3c7bd02251db7763758683c60afc4fb2e3cbb170",
        "p0330009_20100905.nc": "ee5ce7881c74de9d765ae46c6baed4745b0620dd57f7cd7f8b9f1c7f46570836",
        "p0330010_20100905.nc": "fad0d9bb08fc874ffacf8ed1ae25522d48ff9ff8c8f1db82bcaf4e613d6c46e3",
        "p0330011_20100905.nc": "3b3ae89c653651e10b7a8058536d276aae9958e7d4236a4164e05707ef5a8660",
        "p0330012_20100905.nc": "b6318d496ccddd14ede295613a2b4854f0d228c8db996737b5187e73ed226d2a",
        "p0330013_20100906.nc": "efeb37be650368470a6fd7a9cf0cc0b6938ee9eb96b10bde166ef86cda9b6082",
        "p0330014_20100906.nc": "6ab8bea8bc28a1d270de2da3e4cafe9aff4a01a887ebbf7d39bde5980cf585f5",
        "p0330015_20100906.nc": "d22c2bf453601d33e51dc7f3aeecb823fefb532d44b07d45240f5400bc12b464",
        "p0330016_20100906.nc": "404a22da318909c42dcb65fd7315ae1f5542ed8b057faa30c7b21e07f50d67a9",
        "p0330017_20100906.nc": "06df146adee75439cc4f42e27038bec5596705fbe0a93ea53d2d65d94d719504",
        "p0330018_20100906.nc": "3dc2363797adce65c286c5099f0144bb0583b368c84dc24185caba0cad9478a7",
        "p0330019_20100906.nc": "7e37ad465f720ea1f1257830a595677f6d5f85d7391e711d6164ccee8ada5399",
    },
)"""

# Instead of loading from the server, we will load from a local directory for testing and development purposes.
# The local directory will contain the same files as the server, but we will not use pooch to manage them.
# Instead, we will just read them directly from the filesystem.
def load_sample_dataset(file_path: str = str(parent_dir / "data/demo_sg005/p0050001_20080606.nc")) -> xr.Dataset:
    """Download sample datasets for use with seagliderOG1.

    Parameters
    ----------
    dataset_name : str, optional
        Name of the sample dataset to load. Must be one of the available
        datasets in the registry. Default is "p0330015_20100906.nc".

    Returns
    -------
    xarray.Dataset
        The requested sample dataset loaded from the cache.

    Raises
    ------
    KeyError
        If the requested dataset is not available in the registry.

    """
    if os.path.isfile(file_path):
        return xr.open_dataset(file_path, decode_timedelta=False)
    else:
        msg = f"Requested sample dataset {file_path} not known. Available datasets are: {os.listdir(str(parent_dir / 'data/demo_sg005'))}"
        raise KeyError(msg)

def _validate_filename(filename: str) -> bool:
    """Validate if filename matches expected Seaglider basestation patterns.

    Validates against two expected patterns:
    1. p1234567.nc (7 digits after 'p')
    2. p0420100_20100903.nc (7 digits, underscore, 8 digits)

    Also validates that both glider serial number and profile number are positive.

    Parameters
    ----------
    filename : str
        The filename to validate.

    Returns
    -------
    bool
        True if filename matches expected pattern and has valid numbers.

    """
    # pattern 1: p1234567.nc
    pattern1 = r"^p\d{7}\.nc$"
    # pattern 2: p0420100_20100903.nc
    pattern2 = r"^p\d{7}_\d{8}\.nc$"
    if re.match(pattern1, filename) or re.match(pattern2, filename):
        glider_sn = _glider_sn_from_filename(filename)
        divenum = _profnum_from_filename(filename)
        if int(glider_sn) > 0 and int(divenum) > 0:
            return True
        else:
            return False
    else:
        return False


def _profnum_from_filename(filename: str) -> int:
    """Extract the profile/dive number from a Seaglider filename.

    Extracts characters 4-7 (0-indexed) which represent the dive cycle number
    in filenames like p0420001.nc or p0420001_20100903.nc.

    Parameters
    ----------
    filename : str
        Seaglider filename to parse.

    Returns
    -------
    int
        The profile/dive number.

    """
    return int(filename[4:8])


def _glider_sn_from_filename(filename: str) -> int:
    """Extract the glider serial number from a Seaglider filename.

    Extracts characters 1-3 (0-indexed) which represent the 3-digit glider
    serial number in filenames like p0420001.nc.

    Parameters
    ----------
    filename : str
        Seaglider filename to parse.

    Returns
    -------
    int
        The glider serial number.

    """
    return int(filename[1:4])


def filter_files_by_profile(file_list: list[str], start_profile: int | None = None, end_profile: int | None = None) -> list[str]:
    """Filter files by profile/dive number range.

    Filters Seaglider basestation files based on profile number range.
    Expects filenames of the form pXXXYYYY.nc, where XXX is the 3-digit
    glider serial number and YYYY is the 4-digit dive cycle number.

    Example: p0420001.nc represents glider 042, dive 0001.

    Note: Input file_list does not need to be sorted.

    Parameters
    ----------
    file_list : list of str
        List of Seaglider filenames to filter.
    start_profile : int, optional
        Minimum profile number (inclusive).
    end_profile : int, optional
        Maximum profile number (inclusive).

    Returns
    -------
    list of str
        Filtered list of filenames within the specified range.

    """
    filtered_files = []

    file_list = [f for f in file_list if _validate_filename(f)]

    #    divenum_values = [int(file[4:8]) for file in file_list]

    # This could be refactored: see divenum_values above, and find values between start_profile and end_profil
    for file in file_list:
        # Extract the profile number from the filename now from the beginning
        profile_number = _profnum_from_filename(file)
        if start_profile is not None and end_profile is not None:
            if start_profile <= profile_number <= end_profile:
                filtered_files.append(file)
        elif start_profile is not None:
            if profile_number >= start_profile:
                filtered_files.append(file)
        elif end_profile is not None:
            if profile_number <= end_profile:
                filtered_files.append(file)
        else:
            filtered_files.append(file)

    return filtered_files


def load_first_basestation_file(source: str) -> xr.Dataset:
    """Load the first (alphabetically) basestation file from a source.

    Useful for quick examination of data structure and metadata from
    a Seaglider mission without loading all files.

    Parameters
    ----------
    source : str
        URL or local directory path containing NetCDF files.

    Returns
    -------
    xarray.Dataset
        The first basestation dataset.

    """
    file_list = list_files(source)
    filename = file_list[0]
    start_profile = _profnum_from_filename(filename)
    datasets = load_basestation_files(source, start_profile, start_profile)
    return datasets[0]


def load_basestation_files(source: str, start_profile: int | None = None, end_profile: int | None = None) -> list[xr.Dataset]:
    """Load multiple Seaglider basestation files with optional profile filtering.

    Main function for loading Seaglider data from either online repositories
    or local directories. Supports filtering by dive/profile number range.

    Parameters
    ----------
    source : str
        Directory path containing Seaglider NetCDF files.
    start_profile : int, optional
        Minimum profile number to load.
    end_profile : int, optional
        Maximum profile number to load.

    Returns
    -------
    list of xarray.Dataset
        List of loaded basestation datasets, ordered by filename.

    """
    file_list = list_files(source)
    filtered_files = filter_files_by_profile(file_list, start_profile, end_profile)

    datasets = []

    ### Include a tqdm progress bar
    for file in tqdm(filtered_files, desc="Loading datasets", unit="file"):
        ds = xr.open_dataset(os.path.join(source, file), decode_times=False)

        datasets.append(ds)

    return datasets


def list_files(
    source: str, registry_loc: str = "seagliderOG1", registry_name: str = "seaglider_registry.txt"
) -> list[str]:
    """List NetCDF files from a source (URL or local directory).

    For online sources, scrapes directory listings using BeautifulSoup.
    For local sources, lists files in the directory.

    Parameters
    ----------
    source : str
        URL (http/https) or local directory path to scan for files.
    registry_loc : str, optional
        Legacy parameter, not currently used.
    registry_name : str, optional
        Legacy parameter, not currently used.

    Returns
    -------
    list of str
        Sorted list of NetCDF filenames (.nc files only).

    Raises
    ------
    ValueError
        If source is neither a valid URL nor directory path.
    """
    if source.startswith("http://") or source.startswith("https://"):
        # List all files in the URL directory
        response = requests.get(source)
        response.raise_for_status()  # Raise an error for bad status codes

        soup = BeautifulSoup(response.text, "html.parser")
        file_list = []

        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href.endswith(".nc"):
                file_list.append(href)

    elif os.path.isdir(source):
        ### only list files that are nc files
        file_list = [f for f in os.listdir(source) if f.endswith(".nc")]
    else:
        raise ValueError("Source must be a valid URL or directory path.")

    # Sort alphabetically
    file_list.sort()

    return file_list
