"""
compute_dendrograms.py
======================
Computes HCM single-linkage dendrograms for each zone of the asteroid belt
and saves them to disk as Z_zone{N}.npy + zone{N}_df.csv.

Run this on a machine with sufficient RAM.
Once complete, copy the .npy and .csv files to your local machine and use
recut_all_zones() in hcm_clean.ipynb to do analysis without recomputing.

Usage
-----
    # Full run with subsampling for large zones
    python compute_dendrograms.py \
        --proper proper_asteroid_data_clean.csv \
        --families family_membership.csv \
        --outdir output \
        --max-per-zone 80000

    # Skip zones already computed (safe to re-run after a crash)
    python compute_dendrograms.py ... --skip-existing

    # Run specific zones only
    python compute_dendrograms.py ... --zones 3 4 7

Subsampling strategy
--------------------
For zones too large to fit in RAM, we keep ALL known family members
(from family_membership.csv) and randomly sample the background to reach
max-per-zone total. This preserves family recovery while cutting memory.

Rule of thumb for --max-per-zone:
    dist array GB  = N*(N-1)/2 * 4 / 1e9
    linkage peak   = dist array * ~2.5
    safe N for 200 GB available: ~80,000  (-> 13 GB dist, 32 GB linkage)
    safe N for 100 GB available: ~55,000
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

K1, K2, K3  = 5/4, 2, 2      # Zappalà et al. 1990, Section II
AU_YR_TO_MS = 4740.9          # 1 AU/yr in m/s

ZONE_BOUNDARIES = {
    2: (2.065, 2.3),
    3: (2.3,   2.501),
    4: (2.501, 2.825),
    5: (2.825, 2.958),
    6: (2.958, 3.030),
    7: (3.030, 3.278),
}


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def compute_distances(X, chunk_size=1000, verbose=True):
    """
    Compute all N*(N-1)/2 pairwise delta_v distances (Zappalà 1990, Eq. 2).

    For each row i, distances to all j > i are computed simultaneously
    using numpy vectorisation — no Python loop per pair.

    Peak memory ~ dist array = N*(N-1)/2 * 4 bytes (float32).

    Parameters
    ----------
    X          : array (N, 3) — columns [a', e', sin_i']
    chunk_size : rows processed per outer iteration
    verbose    : print progress with ETA

    Returns
    -------
    dist : condensed distance array, length N*(N-1)/2, dtype float32
    """
    N       = len(X)
    n_pairs = N * (N - 1) // 2
    dist    = np.empty(n_pairs, dtype=np.float32)

    a, e, si = X[:, 0], X[:, 1], X[:, 2]
    pair_idx = 0
    t0       = time.time()

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)

        if verbose and i_start % (chunk_size * 20) == 0:
            elapsed = time.time() - t0
            pct     = 100 * pair_idx / n_pairs if n_pairs > 0 else 0
            eta     = (elapsed / pct * (100 - pct)) if pct > 0.1 else float('nan')
            print(f"    {pct:5.1f}%  ({pair_idx:,} / {n_pairs:,} pairs) "
                  f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s",
                  flush=True)

        for i in range(i_start, i_end):
            j = i + 1
            if j >= N:
                continue

            a0  = (a[i] + a[j:]) / 2.0
            n0  = 2.0 * np.pi / (a0 ** 1.5)
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


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_zone(zone_df, max_n, random_state=42):
    """
    Reduce zone_df to max_n rows while keeping ALL known family members.

    Family members (family1 > 0) are always kept — only background
    asteroids (family1 == 0) are randomly subsampled.
    """
    family_members = zone_df[zone_df['family1'] > 0]
    background     = zone_df[zone_df['family1'] == 0]

    n_fam = len(family_members)
    n_bg  = max(0, max_n - n_fam)

    if n_fam > max_n:
        print(f"  WARNING: {n_fam:,} family members exceeds max_n={max_n:,}. "
              f"Keeping all family members, no background.")
        return family_members.reset_index(drop=True)

    if len(background) > n_bg:
        background = background.sample(n=n_bg, random_state=random_state)

    return pd.concat([family_members, background]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main per-zone routine
# ---------------------------------------------------------------------------

def process_zone(df, zone, outdir, max_n=None, chunk_size=1000):
    """
    Compute and save the dendrogram for one zone.

    Saves:
      Z_zone{zone}.npy   — scipy linkage matrix
      zone{zone}_df.csv  — asteroid subset used (after subsampling)
    """
    a_min, a_max = ZONE_BOUNDARIES[zone]

    zone_df = df[
        (df['a'] >= a_min) & (df['a'] < a_max)
    ].dropna(subset=['a', 'e', 'sin_i']).copy()

    n_original = len(zone_df)

    if max_n is not None and n_original > max_n:
        n_fam = (zone_df['family1'] > 0).sum()
        n_bg  = max(0, max_n - n_fam)
        print(f"  Subsampling: {n_original:,} → {max_n:,} "
              f"({n_fam:,} family + {min(n_bg, n_original-n_fam):,} background)")
        zone_df = subsample_zone(zone_df, max_n)

    n       = len(zone_df)
    n_pairs = n * (n - 1) // 2
    dist_gb = n_pairs * 4 / 1e9
    link_gb = dist_gb * 2.5

    print(f"\n{'='*60}")
    print(f"Zone {zone}  ({a_min}–{a_max} AU)")
    print(f"  Asteroids  : {n:,}"
          + (f"  (from {n_original:,})" if n < n_original else ""))
    print(f"  Pairs      : {n_pairs:,}")
    print(f"  Dist array : {dist_gb:.1f} GB")
    print(f"  Linkage est: {link_gb:.1f} GB peak")
    print(f"{'='*60}")

    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        print(f"  Available RAM: {avail:.1f} GB")
        if link_gb > avail * 0.85:
            print(f"  WARNING: may exceed available RAM. "
                  f"Use a smaller --max-per-zone.")
    except ImportError:
        pass

    X = zone_df[['a', 'e', 'sin_i']].values

    print(f"\n  [1/2] Computing pairwise delta_v distances...")
    t0   = time.time()
    dist = compute_distances(X, chunk_size=chunk_size, verbose=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\n  [2/2] Building single-linkage dendrogram...")
    t0 = time.time()
    Z  = linkage(dist, method='single')
    print(f"  Done in {time.time()-t0:.1f}s")

    del dist  # free before saving

    z_path  = os.path.join(outdir, f'Z_zone{zone}.npy')
    df_path = os.path.join(outdir, f'zone{zone}_df.csv')

    np.save(z_path, Z)
    zone_df.to_csv(df_path, index=False)

    print(f"\n  Saved: {z_path}  ({os.path.getsize(z_path)/1e6:.1f} MB)")
    print(f"  Saved: {df_path}  ({os.path.getsize(df_path)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute HCM dendrograms for asteroid belt zones.'
    )
    parser.add_argument('--proper',   required=True,
                        help='Path to proper_asteroid_data_clean.csv')
    parser.add_argument('--families', required=True,
                        help='Path to family_membership.csv')
    parser.add_argument('--outdir',   default='.',
                        help='Output directory (default: .)')
    parser.add_argument('--zones',    nargs='+', type=int,
                        default=list(ZONE_BOUNDARIES.keys()),
                        help='Zones to process (default: 2 3 4 5 6 7)')
    parser.add_argument('--max-per-zone', type=int, default=None,
                        dest='max_per_zone',
                        help='Subsample zones larger than this. '
                             'All known family members are always kept. '
                             'Recommended: 80000 for ~200 GB available RAM.')
    parser.add_argument('--chunk',    type=int, default=1000,
                        help='Chunk size for distance computation (default: 1000)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip zones whose Z_zone*.npy already exists')
    args = parser.parse_args()

    invalid = [z for z in args.zones if z not in ZONE_BOUNDARIES]
    if invalid:
        print(f"ERROR: invalid zones {invalid}. "
              f"Valid: {list(ZONE_BOUNDARIES.keys())}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # Load
    print("Loading data...")
    t0       = time.time()
    proper   = pd.read_csv(args.proper,   low_memory=False)
    families = pd.read_csv(args.families, low_memory=False)
    df = proper.merge(families[['name', 'family1']], on='name', how='left')
    df['family1'] = df['family1'].fillna(0).astype(int)
    print(f"  Loaded {len(df):,} asteroids in {time.time()-t0:.1f}s")

    # Zone size summary
    print(f"\nZone summary (max-per-zone={args.max_per_zone}):")
    print(f"  {'Zone':>6} {'N_total':>10} {'N_family':>10} "
          f"{'N_used':>10} {'Dist GB':>9} {'Link GB':>9}")
    print(f"  {'-'*58}")
    for zone in args.zones:
        a_min, a_max = ZONE_BOUNDARIES[zone]
        zdf     = df[(df['a'] >= a_min) & (df['a'] < a_max)]
        n_total = len(zdf)
        n_fam   = (zdf['family1'] > 0).sum()
        n_use   = min(n_total, args.max_per_zone) if args.max_per_zone else n_total
        pairs   = n_use * (n_use - 1) // 2
        dist_gb = pairs * 4 / 1e9
        link_gb = dist_gb * 2.5
        print(f"  {zone:>6} {n_total:>10,} {n_fam:>10,} "
              f"{n_use:>10,} {dist_gb:>9.1f} {link_gb:>9.1f}")
    print()

    # Process
    total_start = time.time()
    completed, skipped, failed = [], [], []

    for zone in args.zones:
        z_path = os.path.join(args.outdir, f'Z_zone{zone}.npy')

        if args.skip_existing and os.path.exists(z_path):
            print(f"Zone {zone}: skipping — already exists")
            skipped.append(zone)
            continue

        try:
            process_zone(df, zone, args.outdir,
                         max_n=args.max_per_zone,
                         chunk_size=args.chunk)
            completed.append(zone)
        except MemoryError:
            print(f"\nERROR: Zone {zone} — out of memory.")
            print(f"  Try a smaller --max-per-zone value.")
            failed.append(zone)
        except Exception as exc:
            print(f"\nERROR: Zone {zone} — {exc}")
            import traceback; traceback.print_exc()
            failed.append(zone)

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Finished in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Completed : zones {completed}")
    print(f"  Skipped   : zones {skipped}")
    print(f"  Failed    : zones {failed}")
    print(f"\nOutput: {os.path.abspath(args.outdir)}")
    print("Copy Z_zone*.npy and zone*_df.csv to your local machine,")
    print("then run recut_all_zones() in hcm_clean.ipynb.")


if __name__ == '__main__':
    main()