"""Module to compute consensus from six base FLPEs

Runs on a single reach or set of reaches and requires JSON data for reach retrieved by
AWS Batch index.
"""

import argparse as ap
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset, chartostring

ALGO_METADATA = {
    "momma": {"qvar": "Q", "time": "time_str"},
    "metroman": {"qvar": "average/allq", "time": "time_str"},
    "hivdi": {"qvar": "reach/Q", "time": "time_str"},
    "sic4dvar": {"qvar": "Q_da", "time": "times"},
    "busboi": {"qvar": "q/q", "time": "time"},
}

FILL_VALUE = -999999999999.0
FILL_VALUE_STR = "no_data"
CV_THRESH = 0.3


def get_ts(mnt_dir, reach, algo):
    col_map = ALGO_METADATA[algo]

    reach_algo_fp = mnt_dir / "data" / "flpe" / f"{reach}_{algo}.nc"
    if not reach_algo_fp.is_file():
        print(f"FLPE file not found: {reach_algo_fp}")
        return

    with Dataset(reach_algo_fp) as ds:
        raw_t = ds[col_map["time"]][:]

        match algo:
            case "hivdi":
                raw_t_string = chartostring(raw_t)
                t = pd.to_datetime(
                    raw_t_string, format="%Y-%m-%dT%H:%M:%SZ", errors="coerce"
                )
            case "sic4dvar":
                swot_epoch = pd.Timestamp(2000, 1, 1)
                dts = [pd.Timedelta(d, unit="days") for d in raw_t.filled(np.nan)]
                t = pd.DatetimeIndex([swot_epoch + dt for dt in dts])
            case "busboi":
                swot_epoch = pd.Timestamp(2000, 1, 1)
                dts = [pd.Timedelta(d, unit="seconds") for d in raw_t.filled(np.nan)]
                t = pd.DatetimeIndex([swot_epoch + dt for dt in dts])
            case "momma" | "metroman":
                t = pd.to_datetime(raw_t, format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
            case _:
                raise NotImplementedError(
                    f"Algorithm {algo} not implemented in consensus"
                )

        # BUSBOI rounds off the datetimes into just dates so we need to do that for all of them for alignment.
        t_norm = t.normalize()

        Q = ds[col_map["qvar"]][:].filled(np.nan)

    ts = pd.Series(index=t_norm, data=Q).dropna().rename(algo)

    return ts


def get_actual_time(mnt_dir, reach):
    swot_path = mnt_dir / "data" / "input" / "swot" / f"{reach}_SWOT.nc"
    with Dataset(swot_path) as ds:
        time_str = chartostring(ds["reach"]["time_str"][:])
    t = pd.to_datetime(time_str, format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")

    return t


def calc_consensus(df: pd.DataFrame):
    cv = df.std(skipna=True) / df.mean(skipna=True)
    included = cv[cv > CV_THRESH].index.tolist()

    if not included:
        return None, []

    consensus = df[included].median(axis=1, skipna=True)
    consensus.name = "consensus"

    return consensus, included


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
    ts_list = [get_ts(mntdir, reach_id, algo) for algo in ALGO_METADATA.keys()]
    ts_df = pd.concat(ts_list, axis=1)

    consensus_df, included_algos = calc_consensus(ts_df)
    actual_times = get_actual_time(mntdir, reach_id)

    # Since we have to normalize for busboi and drop NaTs for calcs, we reindex on the
    # normalized datetimes from SWOT and then replace the index with the actual datetimes.
    # This fills back in all the missing data that we dropped.
    padded_df = consensus_df.reindex(actual_times.normalize())
    padded_df.index = actual_times


    time_arr = actual_times.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()
    consensus_arr = padded_df.to_numpy()

    # Build nc file
    outdir = mntdir / "flpe" / "consensus"
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    outfile = outdir / f"{reach_id}_consensus.nc"

    with Dataset(outfile, "w", format="NETCDF4") as dsout:
        dsout.n_algos = str(len(included_algos))
        dsout.contributing_algos = included_algos

        # Add consensus Q
        dsout.createDimension("nt", len(consensus_arr))
        consensus_q = dsout.createVariable(
            "consensus_q", "f8", ("nt",), fill_value=FILL_VALUE
        )
        consensus_q.long_name = "consensus discharge"
        consensus_q.short_name = "discharge_consensus"
        consensus_q.tag_basic_expert = "Basic"
        consensus_q.units = "m^3/s"
        consensus_q.valid_min = -10000000.0
        consensus_q.valid_max = 10000000.0
        consensus_q.comment = "Discharge from the consensus discharge algorithm."

        # Add consensus time_str
        consensus_time_str = dsout.createVariable(
            "time_str", str, ("nt",), fill_value="no_data"
        )
        consensus_time_str.long_name = "time (UTC)"
        consensus_time_str.standard_name = "time"
        consensus_time_str.short_name = "time_string"
        consensus_time_str.calendar = "gregorian"
        consensus_time_str.tag_basic_expert = "Basic"
        consensus_time_str.comment = (
            "Time string giving UTC time. The format is YYYY-MM-DDThh:mm:ssZ, "
            "where the Z suffix indicates UTC time."
        )

        # Fill as needed
        consensus_arr_filled = np.where(
            np.isnan(consensus_arr), FILL_VALUE, consensus_arr
        )
        time_arr_filled = [t if isinstance(t, str) else FILL_VALUE_STR for t in time_arr]

        # Write values
        consensus_q[:] = consensus_arr_filled
        consensus_time_str[:] = np.array(time_arr_filled, dtype="O")


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

    with open(reachfile, "r") as fp:
        reaches = json.load(fp)
        reach_ids = [reaches[i]["reach_id"] for i in indices]

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
        print(
            f"cannot parse range string: '{index_str}'. Must be either a single integer, "
            f"a range such as 1-100, or a comma separated list of integers and/or ranges"
        )

    return sorted(list(set(indices)))


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--mntdir", type=str, default="/mnt", help="Mount directory.")
    parser.add_argument("-i", "--index", type=parse_range, required=True)
    parser.add_argument(
        "-r", "--reachfile", type=str, default="reaches.json", help="Reach JSON file."
    )
    args = parser.parse_args()
    run_consensus(Path(args.mntdir), args.index, args.reachfile)
