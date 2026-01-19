import argparse
from zoneinfo import ZoneInfo

import pandas as pd


def _load_index(path):
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Parquet index is not DatetimeIndex")
    return df.index


def _detect_transitions(start, end, tz):
    if start > end:
        return []
    rng = pd.date_range(start=start, end=end, freq="1h", tz=tz)
    if rng.empty:
        return []
    offsets = rng.map(lambda x: x.utcoffset()).tolist()
    transitions = []
    for i in range(1, len(rng)):
        if offsets[i] != offsets[i - 1]:
            transitions.append(rng[i])
    return transitions


def _print_window(index, center, hours=6, label=""):
    start = center - pd.Timedelta(hours=hours)
    end = center + pd.Timedelta(hours=hours)
    window = index[(index >= start) & (index <= end)]
    print(f"\n{label} window {start} -> {end} (count={len(window)})")
    if len(window) > 0:
        print(window[:10])


def main():
    parser = argparse.ArgumentParser(description="Check DST behavior in a parquet time index.")
    parser.add_argument("path", help="Parquet file path")
    parser.add_argument("--tz", default="America/New_York", help="Target timezone for analysis")
    parser.add_argument("--assume-ny", action="store_true", help="If index is naive, localize as America/New_York")
    args = parser.parse_args()

    idx = _load_index(args.path)
    print(f"Index tz: {idx.tz}")
    print(f"Range: {idx.min()} -> {idx.max()} (count={len(idx)})")

    tz = ZoneInfo(args.tz)
    if idx.tz is None:
        if args.assume_ny:
            idx_local = idx.tz_localize(tz)
            print(f"Localized naive index as {args.tz}")
        else:
            idx_local = idx
            print("Index is naive; use --assume-ny to treat as America/New_York")
    else:
        idx_local = idx.tz_convert(tz)
        print(f"Converted index to {args.tz}")

    df = pd.DataFrame({"ts": idx_local})
    df["date"] = df["ts"].dt.date
    df["hour"] = df["ts"].dt.hour

    day_min = df.groupby("date")["hour"].min().value_counts().sort_index()
    day_max = df.groupby("date")["hour"].max().value_counts().sort_index()

    print("\nFirst hour per day (hour -> count):")
    print(day_min.to_string())
    print("\nLast hour per day (hour -> count):")
    print(day_max.to_string())

    start = idx_local.min()
    end = idx_local.max()
    transitions = _detect_transitions(start, end, tz)
    if transitions:
        print(f"\nDetected {len(transitions)} DST transition points:")
        for t in transitions:
            print(f"- {t} offset={t.utcoffset()}")
            _print_window(idx_local, t, label="Data")
    else:
        print("\nNo DST transitions detected in range (or range too small).")

    hours = df["hour"].value_counts().sort_index()
    print("\nHour distribution (0-23 -> count):")
    print(hours.to_string())


if __name__ == "__main__":
    main()
