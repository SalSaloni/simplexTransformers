import torch
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA 
from mess3 import Mess3, build_dataset
from model import SmallTransformer

EPSILONS =[0.05, 0.15, 0.25]

SEQ_LEN=16
D_MODEL=64
N_LAYERS=2
N_HEADS=2
DEVICE="cpu"

def load_model_and_data():
    seqs = np.load("sequences.npy")
    labs = np.load("labels.npy")

    model =SmallTransformer(vocab_size=3, d_model=D_MODEL, n_heads=N_HEADS,n_layers=N_LAYERS, context_len=SEQ_LEN-1, d_ff=128)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model, seqs, labs

def collect_activations(model, sequences, labels, n_samples=300):
    procs = [Mess3(eps) for eps in EPSILONS]

    all_res = {lay:[] for lay in range(N_LAYERS+1)}

    all_bels = []
    all_comp_labs = []
    all_pos = []

    seqs_tensor = torch.tensor(sequences[:n_samples], dtype=torch.long)
    X=seqs_tensor[:,:-1]

    res = model.get_residual_stream(X)

    for idx in range(n_samples):
        seq = sequences[idx]
        comp_id = labels[idx]
        proc = procs[comp_id]

        bels = proc.forward(seq)

        for pos in range(SEQ_LEN-1):
            for lay in range(N_LAYERS+1):
                vec = res[lay][idx, pos, :].numpy()
                all_res[lay].append(vec)

            all_bels.append(bels[pos])
            all_comp_labs.append(comp_id)
            all_pos.append(pos)

        
    for lay in range(N_LAYERS+1):
        all_res[lay] = np.array(all_res[lay])

    return (all_res, np.array(all_bels), np.array(all_comp_labs), np.array(all_pos))


def plot_pca_by_layer(all_residuals, all_comp_labels, all_positions):
    """
    For each layer, plot PCA of residual stream colored by:
    (a) component label
    (b) token position
    """
    n_layers = len(all_residuals)
    fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 9))

    colors_comp = ['#e41a1c', '#377eb8', '#4daf4a']  # red, blue, green
    comp_names = [f'ε={e}' for e in EPSILONS]

    for layer in range(n_layers):
        vecs = all_residuals[layer]  # (N*T, d_model)

        pca = PCA(n_components=2)
        proj = pca.fit_transform(vecs)  # (N*T, 2)
        var = pca.explained_variance_ratio_

        # Top row: color by component
        ax = axes[0, layer]
        for c in range(len(EPSILONS)):
            mask = all_comp_labels == c
            ax.scatter(proj[mask, 0], proj[mask, 1],
                      c=colors_comp[c], label=comp_names[c],
                      alpha=0.4, s=8)
        ax.set_title(f'Layer {layer} — by component\n'
                     f'PC1={var[0]:.2f}, PC2={var[1]:.2f}')
        ax.legend(markerscale=2, fontsize=8)

        # Bottom row: color by position
        ax = axes[1, layer]
        sc = ax.scatter(proj[:, 0], proj[:, 1],
                       c=all_positions, cmap='viridis',
                       alpha=0.4, s=8)
        plt.colorbar(sc, ax=ax, label='position')
        ax.set_title(f'Layer {layer} — by position')

    plt.tight_layout()
    plt.savefig('pca_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pca_by_layer.png")


def plot_belief_simplex(all_residuals, all_beliefs, all_comp_labels, layer=2):
    """
    Project residual stream onto 2D and compare to true belief simplex geometry.
    Color points by their true belief state (distance from simplex corners).
    """
    vecs = all_residuals[layer]
    pca = PCA(n_components=2)
    proj = pca.fit_transform(vecs)

    # Use the dominant belief state as color (argmax of belief vector)
    dominant_state = np.argmax(all_beliefs, axis=1)
    state_colors = ['#e41a1c', '#377eb8', '#4daf4a']

    fig, axes = plt.subplots(1, len(EPSILONS), figsize=(5 * len(EPSILONS), 5))

    for c, (ax, eps) in enumerate(zip(axes, EPSILONS)):
        mask = all_comp_labels == c
        for s in range(3):
            smask = mask & (dominant_state == s)
            ax.scatter(proj[smask, 0], proj[smask, 1],
                      c=state_colors[s], label=f'state {s}',
                      alpha=0.5, s=10)
        ax.set_title(f'Layer {layer}, ε={eps}\ncolored by dominant belief state')
        ax.legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    plt.savefig('belief_simplex.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved belief_simplex.png")


def plot_geometry_vs_position(all_residuals, all_comp_labels, all_positions, layer=2):
    """
    Show how geometry sharpens with context length.
    Early positions (0-3) vs late positions (11-14).
    """
    vecs = all_residuals[layer]
    pca = PCA(n_components=2)
    proj = pca.fit_transform(vecs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors_comp = ['#e41a1c', '#377eb8', '#4daf4a']
    comp_names = [f'ε={e}' for e in EPSILONS]

    for ax, (pos_min, pos_max), title in zip(
        axes,
        [(0, 3), (11, 14)],
        ['Early context (pos 0–3)', 'Late context (pos 11–14)']
    ):
        pos_mask = (all_positions >= pos_min) & (all_positions <= pos_max)
        for c in range(len(EPSILONS)):
            mask = pos_mask & (all_comp_labels == c)
            ax.scatter(proj[mask, 0], proj[mask, 1],
                      c=colors_comp[c], label=comp_names[c],
                      alpha=0.5, s=12)
        ax.set_title(f'Layer {layer} — {title}')
        ax.legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    plt.savefig('geometry_vs_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved geometry_vs_position.png")


if __name__ == "__main__":
    print("Loading model and data...")
    model, sequences, labels = load_model_and_data()

    #Collect activations
    all_residuals, all_beliefs, all_comp_labels, all_positions = \
        collect_activations(model, sequences, labels)

    #plot pca by layer
    plot_pca_by_layer(all_residuals, all_comp_labels, all_positions)

    #plot belief simplex geometry
    plot_belief_simplex(all_residuals, all_beliefs, all_comp_labels, layer=N_LAYERS)

    #plot geometry vs pos
    plot_geometry_vs_position(all_residuals, all_comp_labels, all_positions, layer=N_LAYERS)

