import numpy as np
import pandas as pd

def zappala_distance(ast1, ast2):
    """
    Zappala et al. (1990) velocity metric.
    ast1, ast2: arrays of [a, e, sin_i, n] 
    where n is mean motion in deg/yr
    Returns distance in m/s
    """
    a1, e1, si1, n1 = ast1
    a2, e2, si2, n2 = ast2
    a0 = (a1 + a2) / 2
    n0 = (n1 + n2) / 2

    # Convert n from deg/yr to rad/yr
    n0_rad = n0 * np.pi / 180.0

    da = a1 - a2
    de = e1 - e2
    dsi = si1 - si2

    # Zappala Eq. 2: k1=5/4, k2=2, k3=2
    d = n0_rad * a0 * np.sqrt(
        1.25 * (da / a0)**2 +
        2.0  * de**2 +
        2.0  * dsi**2
    )
    # Convert AU/yr to m/s (1 AU/yr = 1731.46 m/s)
    return d * 1731.46


def hcm(X, seed_idx, d_cutoff, chunk_size=1000):
    """
    Hierarchical Clustering Method.
    Grows a family outward from seed asteroid.
    Never builds the full distance matrix.
    
    X: numpy array [a, e, sin_i, n]
    seed_idx: index of the starting asteroid
    d_cutoff: velocity cutoff in m/s
    """
    family = {seed_idx}
    frontier = {seed_idx}  # newly added members to check

    while frontier:
        new_frontier = set()
        frontier_list = list(frontier)

        # Only compare frontier members against non-members
        non_members = np.array([i for i in range(len(X)) if i not in family])
        if len(non_members) == 0:
            break

        # Process in chunks to control memory
        for start in range(0, len(frontier_list), chunk_size):
            f_chunk = frontier_list[start:start + chunk_size]
            f_data = X[f_chunk]          # shape (chunk, 4)
            nm_data = X[non_members]     # shape (M, 4)

            # Vectorized distance for this chunk only
            for fi, f_ast in zip(f_chunk, f_data):
                a0 = (f_ast[0] + nm_data[:, 0]) / 2
                n0_rad = (f_ast[3] + nm_data[:, 3]) / 2 * np.pi / 180.0
                da  = f_ast[0] - nm_data[:, 0]
                de  = f_ast[1] - nm_data[:, 1]
                dsi = f_ast[2] - nm_data[:, 2]

                dists = n0_rad * a0 * np.sqrt(
                    1.25 * (da / a0)**2 +
                    2.0  * de**2 +
                    2.0  * dsi**2
                ) * 1731.46

                close = non_members[dists < d_cutoff]
                new_frontier.update(close.tolist())

        new_frontier -= family
        family.update(new_frontier)
        frontier = new_frontier

    return family

feature_columns = ['a', 'e', 'sin_i', 'n']
proper = pd.read_csv('proper_asteriod_data_small.csv')
families = pd.read_csv('family_membership.csv')

df = proper.merge(families[['name','family1']], on='name', how='left')
df['family1'] = df['family1'].fillna(0).astype(int)

X = df[feature_columns].dropna().values

D_CUTOFF = 60  # m/s, typical value from literature

assigned = np.full(len(X), -1)  # -1 = unassigned
cluster_id = 0

for seed in range(len(X)):
    if assigned[seed] != -1:
        continue  # already in a family
    
    family = hcm(X, seed, D_CUTOFF)
    
    if len(family) >= 10:  # minimum family size
        for idx in family:
            assigned[idx] = cluster_id
        cluster_id += 1

vesta_idx = df[df['name'] == 4].index[0]
print(f"Vesta at index {vesta_idx}, a={X[vesta_idx, 0]:.3f}")

# Grow just this one family
vesta_family = hcm(X, vesta_idx, D_CUTOFF)
print(f"Vesta family size: {len(vesta_family)}")

# How many does AstDys say the Vesta family has in your dataset?
astdys_vesta = (df['family1'] == 4).sum()
print(f"AstDys Vesta family size in dataset: {astdys_vesta}")

# Quick scatter plot
import matplotlib.pyplot as plt
family_mask = np.array(list(vesta_family))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X[:, 0], X[:, 2], s=2, c='lightgray', label='background')
ax.scatter(X[family_mask, 0], X[family_mask, 2], s=10, c='red', label=f'HCM Vesta ({len(vesta_family)})')
ax.set_xlabel('a (AU)')
ax.set_ylabel('sin(i)')
ax.legend()
plt.title('HCM Vesta Family - Quick Check')
plt.tight_layout()
plt.savefig('vesta_check.png', dpi=120)
plt.show()