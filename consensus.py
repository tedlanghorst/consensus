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
from typing import Literal


ALGO_METADATA = {
    'momma': {
        'qvar': 'Q',
        'time': 'time_str'
    },
    'metroman': {
        'qvar': 'average/allq',
        'time': 'time_str'
    },
    'hivdi': {
        'qvar': 'reach/Q',
        'time': 'time_str'
    },
    'sic4dvar': {
        'qvar': 'Q_da',
        'time': 'times'
    },
    'busboi': {
        'qvar': 'q/q',
        'time': 'time'
    },
}

FILL_VALUE = -999999999999.0
FILL_VALUE_STR = "no_data"
CV_THRESH = 0.3


def remove_low_cv_and_recalc_consensus(arrs, master_times, CV_thresh, included_algos):
    """
    Standardized Anomaly Consensus (Z-Scoring).
    Removes arrays with CV < threshold, converts remaining to Z-scores, computes median,
    and rescales to physical units using the ensemble mean and standard deviation.
    """
    cv_arrs = []
    cv_included_algos = []
    means = []
    stds = []

    for i, arr in enumerate(arrs):

        mean = np.nanmean(arr)
        std = np.nanstd(arr)

        if np.isnan(mean) or mean == 0:
            continue

        cv = std / mean

        if not np.isnan(cv) and cv > CV_thresh:
            cv_arrs.append(arr)
            cv_included_algos.append(included_algos[i])
            means.append(mean)
            stds.append(std)

    if not len(cv_arrs):
        print("All algorithms removed due to low CV; returning NaN array and no included algos.")
        return np.full(len(master_times), np.nan), np.full(len(master_times), FILL_VALUE_STR, dtype=object), []

    z_scores = []
    for arr, mu, sigma in zip(cv_arrs, means, stds):
        z = (arr - mu) / sigma if sigma > 0 else np.full_like(arr, np.nan)
        z_scores.append(z)

    z_consensus = np.nanmedian(np.stack(z_scores, axis=0), axis=0)

    target_mu = np.nanmean(means)
    target_sigma = np.nanmean(stds)

    consensus_arr = (z_consensus * target_sigma) + target_mu
    consensus_arr = np.maximum(consensus_arr, 0.0)

    return consensus_arr, master_times, cv_included_algos

INTERVAL = Literal['s', 'd']
def _num_to_date(algo_time, interval:INTERVAL):
    swot_ts = datetime.datetime(2000, 1, 1)
    new_algo_time = np.full(algo_time.shape[0], FILL_VALUE_STR, dtype=object)

    mask = np.ma.getmaskarray(algo_time)
    valid_indices = np.where(~mask)[0]
    for i in valid_indices:
        t = float(algo_time[i])

        if np.isnan(t):
            continue

        if interval == 's':
            dt = datetime.timedelta(seconds=t)
        elif interval == 'd':
            dt = datetime.timedelta(days=t)
        else:
            raise ValueError(f'Invalid interval: {interval}')

        new_algo_time[i] = (swot_ts + dt).strftime("%Y-%m-%dT%H:%M:%SZ")

    return new_algo_time


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
                arr = ds[metadata['qvar']][:].filled(np.nan)

                if arr.size <= 1:
                    print(f"  Skipping {algo} for reach {reach_id}: array too small (size={arr.size})")
                    continue

                algo_time = ds.variables[metadata['time']][:]
                if algo == 'sic4dvar':
                    algo_time = _num_to_date(algo_time, 'd')
    
                if algo == 'busboi':
                    algo_time = _num_to_date(algo_time, 's')

                time = algo_time

                arr[arr < 0] = np.nan
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

    cleaned_time_arrs = [get_filled_time_arr(t) for t in time_arrs]
    master_times_arr = cleaned_time_arrs[0]

    valid_times = [t for t in master_times_arr if t != FILL_VALUE_STR]
    if not valid_times:
        print(f"No valid times found for reach '{reach_id}'")
        return

    consensus_arr, time_arr, included_algos = remove_low_cv_and_recalc_consensus(
        arrs=arrs, master_times=master_times_arr, CV_thresh=CV_THRESH, included_algos=included_algos
    )

    outdir = mntdir / 'flpe' / 'consensus_z'
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    outfile = outdir / f'{reach_id}_consensus_z.nc'

    with Dataset(outfile, 'w', format="NETCDF4") as dsout:
        dsout.n_algos = str(len(included_algos))
        dsout.contributing_algos = included_algos

        dsout.createDimension("nt", len(consensus_arr))
        consensus_q = dsout.createVariable("consensus_z_q", "f8", ("nt",), fill_value=FILL_VALUE)
        consensus_q.long_name = 'consensus_z discharge'
        consensus_q.short_name = "discharge_consensus_z"
        consensus_q.tag_basic_expert = "Basic"
        consensus_q.units = "m^3/s"
        consensus_q.valid_min = -10000000.0
        consensus_q.valid_max = 10000000.0
        consensus_q.comment = "Discharge from the consensus_z discharge algorithm."

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
        consensus_time_str[:] = time_arr

def get_filled_time_arr(time_arr):

    if hasattr(time_arr, 'ndim') and time_arr.ndim == 2:
        time_arr = chartostring(time_arr)
        
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