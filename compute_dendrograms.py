"""
compute_dendrograms.py

Builds single-linkage HCM dendrograms for each zone of the main asteroid belt.
Reads input CSVs from input_data/, writes Z_zone{N}.npy and zone{N}_df.csv to output_data/.

Usage:
    python compute_dendrograms.py                        # all zones, no H filter
    python compute_dendrograms.py --H_max 16            # only asteroids with H <= 16
    python compute_dendrograms.py --zones 4 7           # specific zones only
    python compute_dendrograms.py --skip-existing       # resume after a crash
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage


# Zappalà et al. 1990, Eq. 2 — metric coefficients
K1, K2, K3 = 5/4, 2, 2

# 1 AU/yr → m/s conversion (used throughout the paper)
AU_YR_TO_MS = 4740.9

# Zone semimajor axis boundaries (AU), from Table I of Zappalà 1990.
# Boundaries correspond to major Kirkwood gaps (mean-motion resonances with Jupiter).
ZONE_BOUNDARIES = {
    2: (2.065, 2.300),
    3: (2.300, 2.501),
    4: (2.501, 2.825),
    5: (2.825, 2.958),
    6: (2.958, 3.030),
    7: (3.030, 3.278),
}

# Default file locations
INPUT_DIR  = 'input_data'
OUTPUT_DIR = 'test_folder'


def compute_distances(X, chunk_size=1000, verbose=True):
    """
    Compute all N*(N-1)/2 pairwise delta_v distances using Eq. 2 of Zappalà 1990.

    For each row i, all pairs (i, j>i) are vectorised in one numpy call,
    so the inner loop is over rows, not individual pairs.
    Peak memory usage is roughly chunk_size * N * 4 bytes (float32).

    X          : (N, 3) array — columns are [a, e, sin_i]
    chunk_size : how many rows to process per outer iteration
    returns    : condensed distance array of length N*(N-1)/2, compatible with scipy linkage
    """
    N       = len(X)
    n_pairs = N * (N - 1) // 2
    dist    = np.empty(n_pairs, dtype=np.float32)

    a, e, si = X[:, 0], X[:, 1], X[:, 2]
    pair_idx = 0
    t0 = time.time()

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)

        if verbose and i_start % (chunk_size * 20) == 0:
            elapsed = time.time() - t0
            pct = 100 * pair_idx / n_pairs if n_pairs else 0
            eta = (elapsed / pct * (100 - pct)) if pct > 0.1 else float('nan')
            print(f"    {pct:5.1f}% ({pair_idx:,}/{n_pairs:,}) "
                  f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s", flush=True)

        for i in range(i_start, i_end):
            j = i + 1
            if j >= N:
                continue

            # Midpoint orbital elements (primed quantities in the paper)
            a0  = (a[i] + a[j:]) / 2.0
            n0  = 2.0 * np.pi / (a0 ** 1.5)   # mean motion via Kepler's third law

            da  = a[i] - a[j:]
            de  = e[i] - e[j:]
            dsi = si[i] - si[j:]

            dv = n0 * a0 * np.sqrt(
                K1 * (da / a0)**2 +
                K2 * de**2 +
                K3 * dsi**2
            ) * AU_YR_TO_MS

            n_new = len(dv)
            dist[pair_idx : pair_idx + n_new] = dv
            pair_idx += n_new

    return dist


def process_zone(df, zone, outdir, max_n=None, chunk_size=1000):
    """
    Run the full pipeline for one zone:
      1. Filter df to the zone's semimajor axis range
      2. Compute all pairwise delta_v distances
      3. Build single-linkage dendrogram
      4. Save Z_zone{zone}.npy and zone{zone}_df.csv to outdir
    """
    a_min, a_max = ZONE_BOUNDARIES[zone]

    zone_df = df[
        (df['a'] >= a_min) &
        (df['a'] <  a_max)
    ].dropna(subset=['a', 'e', 'sin_i']).copy()

    n_original = len(zone_df)

    if max_n and n_original > max_n:
        print(f"  Subsampling {n_original:,} → {max_n:,} (uniform random)")
        zone_df = zone_df.sample(n=max_n, random_state=42).reset_index(drop=True)

    n       = len(zone_df)
    n_pairs = n * (n - 1) // 2
    dist_gb = n_pairs * 4 / 1e9

    print(f"\n{'='*60}")
    print(f"Zone {zone}  ({a_min}–{a_max} AU)")
    print(f"  Asteroids : {n:,}")
    print(f"  Pairs     : {n_pairs:,}  (~{dist_gb:.1f} GB distance array)")
    print(f"{'='*60}")

    X = zone_df[['a', 'e', 'sin_i']].values

    print("\n[1/2] Computing pairwise distances...")
    t0   = time.time()
    dist = compute_distances(X, chunk_size)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n[2/2] Building single-linkage dendrogram...")
    t0 = time.time()
    Z  = linkage(dist, method='single')
    print(f"  Done in {time.time()-t0:.1f}s")

    del dist  # free memory before saving

    z_path  = os.path.join(outdir, f'Z_zone{zone}.npy')
    df_path = os.path.join(outdir, f'zone{zone}_df.csv')

    np.save(z_path, Z)
    zone_df.to_csv(df_path, index=False)

    print(f"\n  Saved: {z_path}")
    print(f"  Saved: {df_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute HCM dendrograms for asteroid belt zones.'
    )

    # Input files default to input_data/
    parser.add_argument(
        '--proper',
        default=os.path.join(INPUT_DIR, 'proper_asteroid_data_clean.csv'),
        help='Path to proper elements CSV (default: input_data/proper_asteroid_data_clean.csv)'
    )
    parser.add_argument(
        '--families',
        default=os.path.join(INPUT_DIR, 'family_membership.csv'),
        help='Path to AstDys family membership CSV (default: input_data/family_membership.csv)'
    )

    # Output goes to output_data/ by default
    parser.add_argument(
        '--outdir',
        default=OUTPUT_DIR,
        help='Directory to write Z_zone*.npy and zone*_df.csv (default: output_data)'
    )

    parser.add_argument(
        '--zones', nargs='+', type=int,
        default=list(ZONE_BOUNDARIES.keys()),
        help='Which zones to process (default: all)'
    )
    parser.add_argument(
        '--H_max', type=float, default=None,
        help='Only include asteroids with H <= this value'
    )
    parser.add_argument(
        '--max-per-zone', type=int, default=None, dest='max_per_zone',
        help='Subsample each zone to at most this many asteroids (for memory-limited runs)'
    )
    parser.add_argument(
        '--chunk', type=int, default=1000,
        help='Rows per chunk in distance computation (tune for available RAM)'
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip zones that already have a saved Z file (useful for resuming)'
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading data from {INPUT_DIR}/...")
    proper   = pd.read_csv(args.proper,   low_memory=False)
    families = pd.read_csv(args.families, low_memory=False)

    # Merge proper elements with family assignments
    df = proper.merge(families[['name', 'family1']], on='name', how='left')
    df['family1'] = df['family1'].fillna(0).astype(int)
    print(f"  {len(df):,} asteroids loaded")

    if args.H_max is not None:
        before = len(df)
        df     = df[df['H_mag'] <= args.H_max].copy()
        print(f"  H <= {args.H_max} filter: {before:,} → {len(df):,}")

    # Print a zone summary before starting
    print(f"\n{'Zone':>6} {'N':>8} {'Fam%':>6} {'DistGB':>8}")
    for zone in args.zones:
        a_min, a_max = ZONE_BOUNDARIES[zone]
        zdf = df[(df['a'] >= a_min) & (df['a'] < a_max)]
        n   = len(zdf)
        n_use = min(n, args.max_per_zone) if args.max_per_zone else n
        fam_pct = 100 * (zdf['family1'] > 0).sum() / max(n, 1)
        dist_gb = n_use * (n_use - 1) // 2 * 4 / 1e9
        print(f"{zone:>6} {n:>8,} {fam_pct:>5.1f}% {dist_gb:>8.1f}")

    print()

    completed, failed = [], []

    for zone in args.zones:
        z_path = os.path.join(args.outdir, f'Z_zone{zone}.npy')

        if args.skip_existing and os.path.exists(z_path):
            print(f"Zone {zone}: already exists, skipping")
            continue

        try:
            process_zone(df, zone, args.outdir,
                         max_n=args.max_per_zone, chunk_size=args.chunk)
            completed.append(zone)
        except MemoryError:
            print(f"Zone {zone}: out of memory — try --max-per-zone 30000")
            failed.append(zone)

    print(f"\nDone.  Completed: {completed}")
    if failed:
        print(f"Failed (OOM): {failed}")


if __name__ == '__main__':
    main()