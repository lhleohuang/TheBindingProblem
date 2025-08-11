import numpy as np
import os
import torch
import random
from datetime import datetime
from vit_prisma.models.model_loader import load_hooked_model
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from PIL import Image
from vit_prisma.utils import prisma_utils
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

model = load_hooked_model('facebook/dino-vitb16', center_writing_weights=True,
                                        fold_ln=True,
                                        refactor_factored_attn_matrices=True,).to("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
])

class ImageLabelDataset(Dataset):
    def __init__(self, images_dir, transform=transform):
        """
        Args:
            images_dir (str): Path to the folder containing PNG images.
            transform (callable, optional): A function/transform to apply to PIL images.
        """
        self.images_dir = images_dir
        self.image_files = sorted([
            f for f in os.listdir(images_dir) if f.lower().endswith('.png')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        fname = self.image_files[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = Image.open(img_path).convert('RGB')

        # Apply transform if provided, else convert to Tensor
        if self.transform:
            img = self.transform(img)
        else:
            arr = np.array(img).astype('float32') / 255.0
            # Convert HWC to CHW
            img = torch.from_numpy(arr.transpose(2, 0, 1))

        # Parse label from filename: "shape1_color1__shape2_color2__idx.png"
        base = os.path.splitext(fname)[0]
        parts = base.split('__')  # ['shape1_color1', 'shape2_color2', 'idx']
        obj1_str, obj2_str, idx_str = parts
        shape1, color1 = obj1_str.split('_')
        shape2, color2 = obj2_str.split('_')
        idx_label = int(idx_str)

        # Label now as ((shape1, color1), (shape2, color2), idx)
        label = ((shape1, color1), (shape2, color2), idx_label)

        return img, label

def collate_fn(batch):
    """Batch images normally, but keep labels as-is."""
    images, labels = zip(*batch)
    images = default_collate(images)
    return images, list(labels)

def get_loader(images_dir, batch_size=32, shuffle=False, num_workers=4, transform=None):
    dataset = ImageLabelDataset(images_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


loader = get_loader("bag_dataset", batch_size=100000) # entire dataset in one batch
test_batch, test_labels = next(iter(loader))

'''
Standard operations using vit-prisma to fetch all activations of all images in the dataset.
'''

all_logits, all_cache = model.run_with_cache(test_batch)

layered_activations = []

for layer in range(model.cfg.n_layers):
    layered_activations.append(all_cache[prisma_utils.get_act_name("resid_mid", layer, None)])

def construct_activation_pairs(layered_activations, test_labels):
    """
    For each layer, constructs:
      - same_pairs: Tensor of shape [N_same, 2, D]
      - diff_pairs: Tensor of shape [N_diff, 2, D]
    based on test_labels entries of form ((s1, c1), (s2, c2), idx).
    
    Args:
      layered_activations: list of L tensors, each [N, T, D]
      test_labels: list of N tuples ((shape1, color1), (shape2, color2), idx)
    
    Returns:
      layered_same, layered_diff: lists of length L with pair tensors
    """
    # Step 1: Extract CLS embeddings per layer
    layered_cls = [layer[:, 0, :] for layer in layered_activations]  # each [N, D]
    
    # Step 2: Build mapping: (s1,c1,s2,c2) -> {idx: image_idx}
    assign_map = {}
    for img_idx, lbl in enumerate(test_labels):
        (s1, c1), (s2, c2), img_idx_lbl = lbl
        key = (s1, c1, s2, c2)
        assign_map.setdefault(key, {})[img_idx_lbl] = img_idx
    
    # Step 3: Find same and diff index pairs
    same_idx = []
    diff_idx = []
    for (s1, c1, s2, c2), idxs in assign_map.items():
        # same: need idx=0 and idx=1
        if 0 in idxs and 1 in idxs:
            same_idx.append((idxs[0], idxs[1]))
        # different: need swapped assignment at idx=0
        swapped = (s1, c2, s2, c1)
        if 0 in idxs and swapped in assign_map and 0 in assign_map[swapped]:
            diff_idx.append((idxs[0], assign_map[swapped][0])) # use only the 0 indexed for the diff pair 
    
    _print_stats(diff_idx, same_idx)

    # Step 4: Construct activation pair tensors
    layered_same = []
    layered_diff = []
    for cls_acts in layered_cls:
        # same pairs
        same_pairs = torch.stack([torch.stack((cls_acts[i], cls_acts[j]), dim=0)
                                   for i, j in same_idx])
        # diff pairs
        diff_pairs = torch.stack([torch.stack((cls_acts[i], cls_acts[j]), dim=0)
                                   for i, j in diff_idx])
        layered_same.append(same_pairs)     # [N_same, 2, D]
        layered_diff.append(diff_pairs)     # [N_diff, 2, D]
    
    return layered_same, layered_diff, assign_map

def construct_activation_pairs_strict_diff(layered_activations, test_labels, use_idx_for_diff=0, balance_to_same=True):
    """
    Build pairs where:
      - SAME: same objects, swapped assignment (idx=0 vs idx=1)
      - DIFF: two images share no shapes and no colors across their two objects
    """
    # 1) CLS per layer
    layered_cls = [layer[:, 0, :] for layer in layered_activations]  # each [N, D]

    # 2) Parse labels & collect vocab
    #    label format: ((shape1, color1), (shape2, color2), idx_label)
    shapes_vocab = set()
    colors_vocab = set()
    img_objs = []  # [( (s1,c1), (s2,c2), idx ), ...]
    for lbl in test_labels:
        (s1, c1), (s2, c2), idx_lbl = lbl
        shapes_vocab.update([s1, s2])
        colors_vocab.update([c1, c2])
        img_objs.append(((s1, c1), (s2, c2), idx_lbl))

    shapes_vocab = sorted(shapes_vocab)
    colors_vocab = sorted(colors_vocab)
    shape2id = {s:i for i,s in enumerate(shapes_vocab)}
    color2id = {c:i for i,c in enumerate(colors_vocab)}

    # 3) SAME pairs (same as before)
    assign_map = {}
    for img_idx, lbl in enumerate(test_labels):
        (s1, c1), (s2, c2), idx_lbl = lbl
        key = (s1, c1, s2, c2)  # ordered
        assign_map.setdefault(key, {})[idx_lbl] = img_idx
    

    same_idx = []
    for key, idxs in assign_map.items():
        if 0 in idxs and 1 in idxs:
            same_idx.append((idxs[0], idxs[1]))
    

    # 4) Build bitmasks for DIFF logic
    #    Also keep one canonical index per unordered object set so we don't duplicate.
    # i.e., only one of ((s1, c1), (s2, c2)) and ((s2, c2), (s1, c1)) will be kept.
    def unordered_key(obj_pair):
        # frozenset of the two (shape,color) tuples
        (s1, c1), (s2, c2) = obj_pair
        return frozenset([(s1, c1), (s2, c2)])

    canonical_img_for_pair = {}  # unordered_key -> img_idx
    shape_masks = []
    color_masks = []
    canonical_indices = []
    

    for img_idx, ((s1, c1), (s2, c2), idx_lbl) in enumerate(img_objs):
        # only for images with idx==use_idx_for_diff (we use the 0 index for idx_lbl)
        if idx_lbl != use_idx_for_diff:
            continue
        key_u = unordered_key(((s1, c1), (s2, c2)))
        # Keep the first occurrence as canonical
        if key_u in canonical_img_for_pair:
            continue
        canonical_img_for_pair[key_u] = img_idx
        # 1 << k means “take the binary number 1 and shift it left by k positions.”
        # Hence 1 << shape2id[s1] is a bitmask where the bit corresponding to shape s1 is set.
        # With '|', if either bit in either number is 1, the result’s bit is 1.
        sm = (1 << shape2id[s1]) | (1 << shape2id[s2]) # shapes present in the image
        cm = (1 << color2id[c1]) | (1 << color2id[c2]) # colors present in the image
        shape_masks.append(sm)
        color_masks.append(cm)
        canonical_indices.append(img_idx)
    

    shape_masks = torch.tensor(shape_masks, dtype=torch.long)
    color_masks = torch.tensor(color_masks, dtype=torch.long)
    # 5) Find disjoint mask pairs (no shared shapes, no shared colors)
    #    We’ll do a simple O(M^2) check on the (likely much smaller) canonical set.
    M = len(canonical_indices)
    diff_idx = []
    
    for i in range(M):
        for j in range(i+1, M):
            # Use bit-wise and '&' == 0 means no overlaps at all
            if (shape_masks[i].item() & shape_masks[j].item()) == 0 and (color_masks[i].item() & color_masks[j].item()) == 0:
                diff_idx.append((canonical_indices[i], canonical_indices[j]))
    

    # Balance DIFF count to SAME count
    if balance_to_same and len(diff_idx) > len(same_idx):
        random.seed(42)  # set seed here if you want reproducibility
        diff_idx = random.sample(diff_idx, len(same_idx))
        
    _print_stats(diff_idx, same_idx)

    # 6) Construct activation pair tensors
    layered_same = []
    layered_diff = []
    
    for cls_acts in layered_cls:
        # SAME
        same_pairs = torch.stack([torch.stack((cls_acts[i], cls_acts[j]), dim=0)
                                  for i, j in same_idx]) if same_idx else torch.empty(0, 2, cls_acts.size(-1))
        # DIFF (strict)
        diff_pairs = torch.stack([torch.stack((cls_acts[i], cls_acts[j]), dim=0)
                                  for i, j in diff_idx]) if diff_idx else torch.empty(0, 2, cls_acts.size(-1))
        layered_same.append(same_pairs)
        layered_diff.append(diff_pairs)

    # For debugging:
    print(f"#SAME pairs: {len(same_idx)}  |  #DIFF(strict) pairs: {len(diff_idx)}  | shapes={len(shapes_vocab)} colors={len(colors_vocab)}")

    return layered_same, layered_diff, assign_map

def construct_activation_pairs_one_fixed_one_changes_one_feature(
    layered_activations, test_labels, use_idx_for_diff=0, balance_to_same=True
):
    """
    Build pairs where:
      - SAME: same objects, swapped assignment (idx=0 vs idx=1)
      - DIFF: between two images, exactly one object is identical (same shape & color),
              and the other object differs in exactly one feature (shape XOR color).
    """
    # 1) CLS per layer
    layered_cls = [layer[:, 0, :] for layer in layered_activations]  # each [N, D]

    # 2) SAME pairs: reuse (idx=0 vs idx=1) per ordered tuple
    assign_map = {}
    for img_idx, lbl in enumerate(test_labels):
        (s1, c1), (s2, c2), idx_lbl = lbl
        key = (s1, c1, s2, c2)  # ordered
        assign_map.setdefault(key, {})[idx_lbl] = img_idx

    same_idx = []
    for key, idxs in assign_map.items():
        if 0 in idxs and 1 in idxs:
            same_idx.append((idxs[0], idxs[1]))

    # 3) Canonicalize images for DIFF construction (only idx==use_idx_for_diff (we use 0 index),
    #    and keep one representative per unordered set of objects)

    # Helper: Parse labels in a convenient form
    # Each item: { "objs": [(s1,c1), (s2,c2)], "idx": idx_label }
    parsed = []
    for lbl in test_labels:
        (s1, c1), (s2, c2), idx_lbl = lbl
        parsed.append({"objs": [(s1, c1), (s2, c2)], "idx": idx_lbl})

    # Helper: unordered key for a 2-object image (ignores order)
    def unordered_key(pair):
        (s1, c1), (s2, c2) = pair
        return frozenset([(s1, c1), (s2, c2)])

    canonical_for_unordered = {}
    canonical_indices = []
    for img_idx, rec in enumerate(parsed):
        if rec["idx"] != use_idx_for_diff:
            continue
        ukey = unordered_key(rec["objs"])
        if ukey in canonical_for_unordered:
            continue
        canonical_for_unordered[ukey] = img_idx
        canonical_indices.append(img_idx)

    # 4) Function to test the "one fixed, one changes exactly one feature" rule
    def one_fixed_one_changes_one_feature(objsA, objsB):
        # objsA, objsB: lists/tuples of two (shape,color) pairs
        setA = set(objsA)
        setB = set(objsB)
        inter = setA & setB
        if len(inter) != 1:
            return False  # need exactly one identical object across images
        fixed = next(iter(inter))
        # Identify the "other" objects
        otherA = next(o for o in objsA if o != fixed)
        otherB = next(o for o in objsB if o != fixed)
        sA, cA = otherA
        sB, cB = otherB
        same_shape = (sA == sB)
        same_color = (cA == cB)
        # exactly-one-feature match (XOR): True iff one matches and the other doesn't
        return (same_shape != same_color)

    # 5) Build DIFF pairs by scanning canonical set (O(M^2))
    diff_idx = []
    M = len(canonical_indices)
    # Pre-extract the object pairs for speed
    objs_by_idx = [parsed[i]["objs"] for i in range(len(parsed))]
    for a_pos in range(M):
        ia = canonical_indices[a_pos]
        objsA = objs_by_idx[ia]
        for b_pos in range(a_pos + 1, M):
            ib = canonical_indices[b_pos]
            objsB = objs_by_idx[ib]
            if one_fixed_one_changes_one_feature(objsA, objsB):
                diff_idx.append((ia, ib))
    # breakpoint()

    # 6) Balance DIFF count to SAME count
    if balance_to_same and len(diff_idx) > len(same_idx):
        random.seed(42)  # set seed here if you want reproducibility
        diff_idx = random.sample(diff_idx, len(same_idx))
    # if balance_to_same and len(diff_idx) > len(same_idx):
    #         diff_idx = diff_idx[:len(same_idx)]
    
    _print_stats(diff_idx, same_idx)

    # 7) Construct activation pair tensors
    layered_same = []
    layered_diff = []
    for cls_acts in layered_cls:
        D = cls_acts.size(-1)
        if same_idx:
            same_pairs = torch.stack(
                [torch.stack((cls_acts[i], cls_acts[j]), dim=0) for i, j in same_idx]
            )
        else:
            same_pairs = torch.empty(0, 2, D)
        if diff_idx:
            diff_pairs = torch.stack(
                [torch.stack((cls_acts[i], cls_acts[j]), dim=0) for i, j in diff_idx]
            )
        else:
            diff_pairs = torch.empty(0, 2, D)
        layered_same.append(same_pairs)
        layered_diff.append(diff_pairs)

    # For debugging:
    print(f"#SAME: {len(same_idx)} | #DIFF(one-fixed-one-change): {len(diff_idx)}")
    # breakpoint()
    return layered_same, layered_diff, assign_map

# Helper for plotting stats
def _bar_triptych(cnt_shapes, cnt_colors, cnt_combo, title, save_dir=None):
    def kv(cnt):
        items = sorted(cnt.items(), key=lambda kv: (-kv[1], str(kv[0])))
        x = [str(k) for k,_ in items]
        y = [v for _,v in items]
        return x, y

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    xs, ys = kv(cnt_shapes)
    axes[0].bar(range(len(xs)), ys)
    axes[0].set_title("Shapes")
    axes[0].set_xticks(range(len(xs)))
    axes[0].set_xticklabels(xs, rotation=60, ha="right")

    xc, yc = kv(cnt_colors)
    axes[1].bar(range(len(xc)), yc)
    axes[1].set_title("Colors")
    axes[1].set_xticks(range(len(xc)))
    axes[1].set_xticklabels(xc, rotation=60, ha="right")

    xk, yk = kv(cnt_combo)
    axes[2].bar(range(len(xk)), yk)
    axes[2].set_title("Shape+Color")
    axes[2].set_xticks(range(len(xk)))
    axes[2].set_xticklabels(xk, rotation=60, ha="right")

    fig.suptitle(title)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        p = Path(save_dir) / f"{title.lower().replace(' ','_').replace('+','plus')}.png"
        fig.savefig(p, dpi=200)
    plt.show()

def _print_stats(diff_idx, same_idx):

    parsed = []
    for lbl in test_labels:
        (s1, c1), (s2, c2), idx_lbl = lbl
        parsed.append({"objs": [(s1, c1), (s2, c2)], "idx": idx_lbl})

    c_shape_diff = Counter()
    c_shape_same = Counter()
    c_color_diff = Counter()
    c_color_same = Counter()
    c_combo_diff = Counter()
    c_combo_same = Counter()

    for i, j in diff_idx:
        for s, c in (parsed[i]["objs"] + parsed[j]["objs"]):
            c_shape_diff[s] += 1
            c_color_diff[c] += 1
            c_combo_diff[(s, c)] += 1

    for i, j in same_idx:
        for s, c in (parsed[i]["objs"] + parsed[j]["objs"]):
            c_shape_same[s] += 1
            c_color_same[c] += 1
            c_combo_same[(s, c)] += 1

    def dump(title, cnt):
        total = sum(cnt.values())
        print(f"\n{title} (total={total}, unique={len(cnt)}):")
        for k, v in sorted(cnt.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            print(f"  {k}: {v}")

    print(f"\n SAME Stats over {len(same_idx)} pairs:")
    dump("Shapes", c_shape_same)
    dump("Colors", c_color_same)
    dump("Shape+Color", c_combo_same)

    print(f"\n")

    print(f"\n DIFF Stats over {len(diff_idx)} pairs:")
    dump("Shapes", c_shape_diff)
    dump("Colors", c_color_diff)
    dump("Shape+Color", c_combo_diff)

    print(f"Combined Stats over {len(same_idx) + len(diff_idx)} pairs:")
    c_shape_combined = c_shape_same + c_shape_diff
    c_color_combined = c_color_same + c_color_diff
    c_combo_combined = c_combo_same + c_combo_diff
    dump("Shapes", c_shape_combined)
    dump("Colors", c_color_combined)
    dump("Shape+Color", c_combo_combined)

    _bar_triptych(c_shape_same, c_color_same, c_combo_same, f"{MODE} SAME Stats", plot_save_dir)
    _bar_triptych(c_shape_diff, c_color_diff, c_combo_diff, f"{MODE} DIFF Stats", plot_save_dir)
    _bar_triptych(c_shape_combined, c_color_combined, c_combo_combined, f"{MODE} COMBINED Stats", plot_save_dir)

def combine_and_shuffle_pairs(layered_same, layered_diff, seed=None):
    """
    Combines same and diff pairs for each layer, labels them, and shuffles.

    Args:
        layered_same: list of L tensors, each [N_same, 2, D]
        layered_diff: list of L tensors, each [N_diff, 2, D]
        seed: optional int for reproducibility

    Returns:
        layered_pairs: list of L tensors, each [N_total, 2, D]
        layered_labels: list of L tensors, each [N_total]
    """
    if seed is not None:
        torch.manual_seed(seed)

    layered_pairs = []
    layered_labels = []
    for same_pairs, diff_pairs in zip(layered_same, layered_diff):
        # Create labels
        n_same = same_pairs.size(0)
        n_diff = diff_pairs.size(0)
        labels_same = torch.zeros(n_same, dtype=torch.long)
        labels_diff = torch.ones(n_diff, dtype=torch.long)

        # Combine pairs and labels
        pairs = torch.cat([same_pairs, diff_pairs], dim=0)  # [N_total, 2, D]
        labels = torch.cat([labels_same, labels_diff], dim=0)  # [N_total]

        # Shuffle
        perm = torch.randperm(pairs.size(0))
        pairs_shuffled = pairs[perm]
        labels_shuffled = labels[perm]

        layered_pairs.append(pairs_shuffled)
        layered_labels.append(labels_shuffled)

    return layered_pairs, layered_labels

def save_pairs(path, layered_pairs, layered_labels, meta=None, fp16=False):
    pairs_cpu  = [p.half().cpu() if fp16 else p.cpu() for p in layered_pairs]
    labels_cpu = [y.cpu() for y in layered_labels]
    payload = {
        "version": 1,
        "timestamp": datetime.utcnow().isoformat(),
        "pairs": pairs_cpu,
        "labels": labels_cpu,
        "meta": meta or {}
    }
    torch.save(payload, path)


if __name__ == "__main__":
    # you may call construct_activation_pairs (for superposition catastrophe style datset) 
    # or construct_activation_pairs_one_fixed_one_changes_one_feature (for that baseline)
    # or construct_activation_pairs_strict_diff (for that baseline)
    MODE = "superposition_catastrophe"  # or "one_fixed_one_changes_one_feature" or "strict_diff"
    plot_save_dir = "dataset_stats_plots"

    if MODE == "superposition_catastrophe":
        layered_same, layered_diff, assign_map = construct_activation_pairs(layered_activations, test_labels)
    elif MODE == "one_fixed_one_changes_one_feature":
        layered_same, layered_diff, assign_map = construct_activation_pairs_one_fixed_one_changes_one_feature(layered_activations, test_labels)
    elif MODE == "strict_diff":
        layered_same, layered_diff, assign_map = construct_activation_pairs_strict_diff(layered_activations, test_labels)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
    
    layered_pairs, layered_labels = combine_and_shuffle_pairs(layered_same, layered_diff, seed=42)
    meta = {"model":"facebook/dino-vitb16", "pair_build":"cls_sum", "seed":42}

    # change saved activations accordingly
    save_pairs(f"layered_pairs_labels_{MODE}.pt", layered_pairs, layered_labels, meta=meta, fp16=False)