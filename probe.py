import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from analyze import load_model_and_data, collect_activations
from mess3 import Mess3

EPSILONS = [0.05, 0.15, 0.25 ]
N_LAYERS=2

def probe_belief_states(all_res, all_bels):
    res = {}
    n=len(all_bels)

    idx_train, idx_test=train_test_split(np.arange(n), test_size=0.2, random_state=42)

    for lay in range(N_LAYERS+1):
        vecs=all_res[lay]

        X_train=vecs[idx_train]
        X_test=vecs[idx_test]

        y_train=all_bels[idx_train]
        y_test = all_bels[idx_test]

        probe=Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)

        r2=r2_score(y_test,y_pred)
        res[lay] = r2

    return res

def probe_by_component(all_res, all_bel, all_comp_labs):
    fig, axes = plt.subplots(1, len(EPSILONS), figsize=(5 * len(EPSILONS), 4))

    for c, (ax, eps) in enumerate(zip(axes, EPSILONS)):
        mask = all_comp_labs == c
        r2_per_layer = []

        idx = np.where(mask)[0]
        idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

        for layer in range(N_LAYERS + 1):
            vecs = all_res[layer]
            probe = Ridge(alpha=1.0)
            probe.fit(vecs[idx_train], all_bel[idx_train])
            y_pred = probe.predict(vecs[idx_test])
            r2 = r2_score(all_bel[idx_test], y_pred)
            r2_per_layer.append(r2)

        ax.plot(range(N_LAYERS + 1), r2_per_layer, 'o-', linewidth=2)
        ax.set_title(f'ε={eps}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('R² (belief state decoding)')
        ax.set_ylim(0, 1)
        ax.set_xticks(range(N_LAYERS + 1))

    plt.suptitle('Linear probe: belief state R² per layer per component', y=1.02)
    plt.tight_layout()
    plt.savefig('probe_by_component.png', dpi=150, bbox_inches='tight')
    plt.close()


def probe_vs_position(all_res, all_bel, all_positions):

    layer = N_LAYERS
    vecs = all_res[layer]

    position_buckets = [(0, 2), (3, 5), (6, 9), (10, 14)]
    r2_per_bucket = []
    bucket_labels = []

    for (pmin, pmax) in position_buckets:
        mask = (all_positions >= pmin) & (all_positions <= pmax)
        idx = np.where(mask)[0]
        if len(idx) < 50:
            continue

        idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
        probe = Ridge(alpha=1.0)
        probe.fit(vecs[idx_train], all_bel[idx_train])
        y_pred = probe.predict(vecs[idx_test])
        r2 = r2_score(all_bel[idx_test], y_pred)
        r2_per_bucket.append(r2)
        bucket_labels.append(f'{pmin}–{pmax}')

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(r2_per_bucket)), r2_per_bucket, 'o-', linewidth=2)
    plt.xticks(range(len(bucket_labels)), bucket_labels)
    plt.xlabel('Position range')
    plt.ylabel('R² (belief state decoding)')
    plt.title('Belief state linear decodability vs context length\n(final layer)')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('probe vs pos', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    model, sequences, labels = load_model_and_data()
    all_res, all_bel, all_comp_labs, all_pos = collect_activations(model, sequences, labels)

    results = probe_belief_states(all_res, all_bel)
    probe_by_component(all_res, all_bel, all_comp_labs)
    probe_vs_position(all_res, all_bel, all_pos)
