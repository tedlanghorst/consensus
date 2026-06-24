"""Module to compute consensus from six base FLPEs

Runs on a single reach or set of reaches and requires JSON data for reach retrieved by
AWS Batch index.
"""

import argparse as ap
from pathlib import Path
import json
from netCDF4 import Dataset, chartostring
import numpy as np
import os
import datetime

ALGO_METADATA = {
    'momma': {
        'qvar':'Q',
        'time':'time_str'
    },
    'metroman':{
        'qvar':'average/allq',
        'time':'time_str'
    },
    'hivdi': {
        'qvar':'reach/Q',
        'time':'time_str'
    },
    'sic4dvar':{
        'qvar':'Q_da',
        'time':'times'
    },
    'busboi':{
        'qvar':'q/q',
        'time':'time'
    },
}
# removing sad for version 4
#    'sad':{
#        'qvar':'Qa',
#        'time':'time_str'
#    },

FILL_VALUE = -999999999999.0
FILL_VALUE_STR = "no_data"
CV_THRESH = 0.3


def remove_low_cv_and_recalc_consensus(arrs, time_arrs, CV_thresh, included_algos):
    """
    For a list of discharge arrays:
    - Removes arrays with CV < threshold
    - Recalculates consensus using the remaining arrays

    Parameters
    ----------
    arrs : list of np.ndarray
        Discharge arrays from each algorithm.
    CV_thresh : float
        Coefficient of variation threshold below which arrays are excluded.

    Returns
    -------
    np.ndarray
        Cleaned and recalculated consensus array.
    """

    cv_arrs = []
    cv_included_algos = []
    cv_time_arrs = []

    for i, arr in enumerate(arrs):
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        cv = std / mean if mean != 0 else np.nan

        if not np.isnan(cv) and cv > CV_thresh:
            cv_arrs.append(arr)
            cv_included_algos.append(included_algos[i])
            cv_time_arrs.append(time_arrs[i])

    if not len(cv_arrs):
        print("All algorithms removed due to low CV; returning NaN array and no included algos.")
        return np.full_like(arrs[0], np.nan), np.full_like(arrs[0], "no_data", dtype=object), []

    # Compute median consensus
    consensus_arr = np.nanmedian(np.stack(cv_arrs, axis=0), axis=0)
    selected_time_arr = time_arrs[0]

    return consensus_arr, selected_time_arr, cv_included_algos


def process_reach(reach_id, mntdir):
    """
    Compute consensus for a single reach.

    Parameters
    ----------
    mntdir: Path
        path to base mount directory
    reach_id: int
        ID of reach to process
    """

    print('reach', reach_id)
    included_algos = []
    arrs = []
    time_arrs = []

    for algo, metadata in ALGO_METADATA.items():
        infile = mntdir / 'flpe' / algo / f'{reach_id}_{algo}.nc'
        if not os.path.exists(infile):
            continue
        try:
            with Dataset(infile, 'r') as ds:
                try: 
                    arr = ds[metadata['qvar']][:].filled(np.nan)
                except KeyError:
                    print(f"Q data variable ({metadata['qvar']}) not found in {infile}")
                    continue

                # Skip if array is effectively empty (e.g. busboi all-NA case returns 1x1)
                if arr.size <= 1:
                    print(f"  Skipping {algo} for reach {reach_id}: array too small (size={arr.size})")
                    continue

                algo_time = ds.variables[metadata['time']][:]
                if algo == 'sic4dvar':
                    mask = np.ma.getmaskarray(algo_time)
                    valid_indexes = [i for i in range(algo_time.shape[0])]

                    if valid_indexes:
                        valid_sic_str = [algo_time[i] for i in valid_indexes]
                        swot_ts = datetime.datetime(2000, 1, 1, 0, 0, 0)

                        valid_sic_str = np.array(valid_sic_str, dtype=float)
                        valid_sic_str = np.where(np.ma.getmaskarray(valid_sic_str), np.nan, valid_sic_str)

                        algo_time = np.array([
                            (swot_ts + datetime.timedelta(days=t)).strftime("%Y-%m-%dT%H:%M:%SZ") if not np.isnan(t) else None
                            for t in valid_sic_str
                        ])

                time = algo_time

                # treat negative discharge as NaN
                arr[arr < 0] = np.nan
                # ignore algos with no nonnegative discharge
                if not np.any(arr >= 0):
                    continue

                arrs.append(arr)
                time_arrs.append(time)
                included_algos.append(algo)

        except (IOError, OSError):
            continue

    if not len(arrs):
        print(f"No data for reach '{reach_id}'")
        return

    # Ensure all arrays are the same length — drop any that don't match the majority
    if len(arrs) > 1:
        lengths = [len(a) for a in arrs]
        most_common_len = max(set(lengths), key=lengths.count)
        keep = [i for i, a in enumerate(arrs) if len(a) == most_common_len]
        if len(keep) < len(arrs):
            dropped = [included_algos[i] for i in range(len(arrs)) if i not in keep]
            print(f"  Dropping {dropped} for reach {reach_id}: length mismatch")
            arrs           = [arrs[i] for i in keep]
            time_arrs      = [time_arrs[i] for i in keep]
            included_algos = [included_algos[i] for i in keep]

    consensus_arr, time_arr, included_algos = remove_low_cv_and_recalc_consensus(
        arrs=arrs, time_arrs=time_arrs, CV_thresh=CV_THRESH, included_algos=included_algos
    )

    # Build nc file
    outdir = mntdir / 'flpe' / 'consensus'
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    outfile = outdir / f'{reach_id}_consensus.nc'

    with Dataset(outfile, 'w', format="NETCDF4") as dsout:
        dsout.n_algos = str(len(included_algos))
        dsout.contributing_algos = included_algos

        # Add consensus Q
        dsout.createDimension("nt", len(consensus_arr))
        consensus_q = dsout.createVariable("consensus_q", "f8", ("nt",), fill_value=FILL_VALUE)
        consensus_q.long_name = 'consensus discharge'
        consensus_q.short_name = "discharge_consensus"
        consensus_q.tag_basic_expert = "Basic"
        consensus_q.units = "m^3/s"
        consensus_q.valid_min = -10000000.0
        consensus_q.valid_max = 10000000.0
        consensus_q.comment = "Discharge from the consensus discharge algorithm."

        # Add consensus time_str
        consensus_time_str = dsout.createVariable("time_str", str, ("nt",), fill_value="no_data")
        consensus_time_str.long_name = "time (UTC)"
        consensus_time_str.standard_name = "time"
        consensus_time_str.short_name = "time_string"
        consensus_time_str.calendar = "gregorian"
        consensus_time_str.tag_basic_expert = "Basic"
        consensus_time_str.comment = (
            "Time string giving UTC time. The format is YYYY-MM-DDThh:mm:ssZ, "
            "where the Z suffix indicates UTC time."
        )

        consensus_q[:] = np.where(np.isnan(consensus_arr), FILL_VALUE, consensus_arr)
        consensus_time_str[:] = get_filled_time_arr(time_arr)


def get_filled_time_arr(time_arr):
    if hasattr(time_arr, 'ndim') and time_arr.ndim == 2:
        time_arr =  chartostring(time_arr)
        
    time_arr_filled = []
    for t in time_arr:
        if t is None or np.ma.is_masked(t):
            time_arr_filled.append(FILL_VALUE_STR)
            continue
            
        decoded_str = t.decode('utf-8') if hasattr(t, 'decode') else str(t)
        cleaned_str = decoded_str.strip()
        
        if cleaned_str in ('NaT', ''):
            time_arr_filled.append(FILL_VALUE_STR)
        else:
            time_arr_filled.append(cleaned_str)

    return np.array(time_arr_filled, dtype=object)


def run_consensus(mntdir, indices, reachfile):
    """
    Run consensus algorithm on a set of reaches.

    Parameters
    ----------
    mntdir: Path
        path to base mount directory
    indices: list
        offsets of reaches to process
    """

    with open(reachfile, 'r') as fp:
        reaches = json.load(fp)
        reach_ids = [reaches[i]['reach_id'] for i in indices]

    for reach_id in reach_ids:
        process_reach(reach_id, mntdir)


def parse_range(index_str):
    """Parse a range string into a list of integers."""

    indices = []
    try:
        for part in index_str.strip().split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                indices.extend(list(range(int(start), int(end) + 1)))
            else:
                indices.append(int(part))
    except (IndexError, ValueError, TypeError):
        print(f"cannot parse range string: '{index_str}'. Must be either a single integer, "
              f"a range such as 1-100, or a comma separated list of integers and/or ranges")

    return sorted(list(set(indices)))


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--mntdir", type=str, default="/mnt", help="Mount directory.")
    parser.add_argument("-i", "--index", type=parse_range, required=True)
    parser.add_argument("-r", "--reachfile", type=str, default="reaches.json", help="Reach JSON file.")
    args = parser.parse_args()
    run_consensus(Path(args.mntdir), args.index, args.reachfile)
