import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os

@torch.no_grad()
def dot_scores(pairs, batch_size=512):
    dl = DataLoader(TensorDataset(pairs), batch_size=batch_size, shuffle=False)
    scores = []
    for (xb,) in dl:
        a, b = xb[:, 0, :], xb[:, 1, :]
        s = (a * b).sum(dim=1)  # raw dot product
        scores.append(s.cpu())
    return torch.cat(scores, dim=0).numpy()

def evaluate_no_training(layered_pairs, layered_labels, batch_size=512):
    aucs = []
    for pairs, labels in zip(layered_pairs, layered_labels):
        scores = dot_scores(pairs, batch_size=batch_size)
        y = labels.float().cpu().numpy()
        y_pos = 1.0 - y  # "same" as positive
        auc = roc_auc_score(y_pos, scores)
        aucs.append(auc)
    return aucs

def load_pairs(path, device="cpu"):
    obj = torch.load(path, map_location="cpu")
    pairs  = [p.to(device) for p in obj["pairs"]]
    labels = [y.to(device) for y in obj["labels"]]
    return pairs, labels, obj.get("meta", {})

# -------- Multiple files plotting --------
saved_activation_files = [
    "layered_pairs_labels_strict_diff.pt",
    "layered_pairs_labels_one_fixed_one_changes_one_feature.pt",
    "layered_pairs_labels_location_stable_0.pt",
    "layered_pairs_labels_location_stable_1.pt",
    "layered_pairs_labels_superposition_catastrophe.pt",
]

os.makedirs("dotproduct_no_training_plots", exist_ok=True)

plt.figure(figsize=(8, 5))
for fname in saved_activation_files:
    layered_pairs, layered_labels, _ = load_pairs(fname)
    aucs = evaluate_no_training(layered_pairs, layered_labels)
    # Extract label from filename
    label = fname[len("layered_pairs_labels_"):-len(".pt")]
    plt.plot(range(len(aucs)), aucs, marker="o", label=label)

plt.title("ROC–AUC per Layer (No Projection)")
plt.xlabel("Layer")
plt.ylabel("ROC–AUC")
plt.ylim(0.0, 1.0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=8)
out_path = "dotproduct_no_training_plots/all_files_roc_auc_per_layer.png"
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved combined plot to {out_path}")
