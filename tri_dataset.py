import os, json, random, math
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw

# ---------- Defaults (edit as you like) ----------
DEFAULT_SHAPES = ["circle", "square", "triangle", "star", "pentagon"]
DEFAULT_COLORS = {
    "red":    (220,  50,  47),
    "blue":   ( 38, 139, 210),
    "green":  (133, 153,  0),
    "yellow": (181, 137,   0),
    "purple": (108, 113, 196),
}
CANVAS = (224, 224)
RADIUS = 25

# ---------- Geometry helpers ----------
def _rand_pos(w=CANVAS[0], h=CANVAS[1], r=RADIUS):
    return random.randint(r, w - r), random.randint(r, h - r)

def _non_overlapping_positions(n=3, w=CANVAS[0], h=CANVAS[1], r=RADIUS, max_tries=10_000):
    pts = []
    thr2 = (2 * r) ** 2
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        x, y = _rand_pos(w, h, r)
        ok = True
        for (px, py) in pts:
            if (x - px) ** 2 + (y - py) ** 2 < thr2:
                ok = False
                break
        if ok:
            pts.append((x, y))
    if len(pts) != n:
        raise RuntimeError("Failed to place non-overlapping objects; try a smaller radius or fewer objects.")
    return pts

# ---------- Shape drawing ----------
def _draw_circle(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], r: int, fill: Tuple[int,int,int]):
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)

def _draw_square(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], r: int, fill: Tuple[int,int,int]):
    x, y = xy
    draw.rectangle((x - r, y - r, x + r, y + r), fill=fill)

def _regular_polygon(cx, cy, r, k, rot=0.0):
    pts = []
    for i in range(k):
        a = rot + 2 * math.pi * i / k
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts

def _draw_triangle(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], r: int, fill: Tuple[int,int,int]):
    x, y = xy
    pts = _regular_polygon(x, y, r, 3, rot=-math.pi/2)
    draw.polygon(pts, fill=fill)

def _draw_pentagon(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], r: int, fill: Tuple[int,int,int]):
    x, y = xy
    pts = _regular_polygon(x, y, r, 5, rot=-math.pi/2)
    draw.polygon(pts, fill=fill)

def _draw_star(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], r: int, fill: Tuple[int,int,int]):
    # 5-point star: alternate between outer and inner radius
    x, y = xy
    outer = r
    inner = r * 0.5
    pts = []
    for i in range(10):
        rr = outer if i % 2 == 0 else inner
        a = -math.pi/2 + i * (math.pi / 5)
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    draw.polygon(pts, fill=fill)

def _draw_shape(draw, shape: str, xy, r, fill):
    if shape == "circle":
        _draw_circle(draw, xy, r, fill)
    elif shape == "square":
        _draw_square(draw, xy, r, fill)
    elif shape == "triangle":
        _draw_triangle(draw, xy, r, fill)
    elif shape == "star":
        _draw_star(draw, xy, r, fill)
    elif shape == "pentagon":
        _draw_pentagon(draw, xy, r, fill)
    else:
        raise ValueError(f"Unknown shape: {shape}")

# ---------- Pair logic ----------
def _random_bag(shapes: List[str], colors: List[str]) -> Tuple[List[str], List[str]]:
    sh = random.sample(shapes, 3)
    co = random.sample(colors, 3)
    return sh, co

def _pairings_for_image(shapes3: List[str], colors3: List[str]) -> List[Tuple[str, str]]:
    cols_perm = colors3[:]
    random.shuffle(cols_perm)
    return list(zip(shapes3, cols_perm))

def _three_cycle_perm(seq: List[str]) -> List[str]:
    # produce a 3-cycle derangement of length 3 (no fixed points)
    # two 3-cycles: (0->1->2->0) or (0->2->1->0)
    if len(seq) != 3:
        raise ValueError("three-cycle only defined for length 3.")
    if random.random() < 0.5:
        order = [1, 2, 0]
    else:
        order = [2, 0, 1]
    return [seq[i] for i in order]

def _diff_pair_repair(shapes3: List[str], colors3_img0: List[str]) -> List[str]:
    # take the exact color set from image0 but apply a 3-cycle so all pairings change
    return _three_cycle_perm(colors3_img0)

def _compose_objects(pairings: List[Tuple[str,str]], positions: List[Tuple[int,int]], color_map: Dict[str,Tuple[int,int,int]]):
    objs = []
    for (shape, color), (x, y) in zip(pairings, positions):
        objs.append({
            "shape": shape,
            "color": color,
            "xy": (x, y),
            "rgb": color_map[color],
        })
    return objs

def _render_image(objects, canvas=CANVAS, bg=(255,255,255), radius=RADIUS):
    img = Image.new("RGB", canvas, bg)
    draw = ImageDraw.Draw(img)
    for o in objects:
        _draw_shape(draw, o["shape"], o["xy"], radius, o["rgb"])
    return img

# ---------- Splitting ----------
def _split_indices(n_items: int, splits=(0.6, 0.2, 0.2), seed=1):
    assert abs(sum(splits) - 1.0) < 1e-6
    idx = list(range(n_items))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_train = int(round(splits[0] * n_items))
    n_val = int(round(splits[1] * n_items))
    n_test = n_items - n_train - n_val
    train = idx[:n_train]
    val = idx[n_train:n_train+n_val]
    test = idx[n_train+n_val:]
    return train, val, test

# ---------- Main generator ----------
def generate_dataset(
    out_dir: str = "bag3_dataset",
    shapes: List[str] = None,
    colors: Dict[str, Tuple[int,int,int]] = None,
    n_same_pairs: int = 400,
    n_diff_pairs: int = 400,
    radius: int = RADIUS,
    splits: Tuple[float,float,float] = (0.6, 0.2, 0.2),
    seed: int = 1
):
    if shapes is None: shapes = DEFAULT_SHAPES
    if colors is None: colors = DEFAULT_COLORS
    random.seed(seed)

    root = Path(out_dir)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "val").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    total_pairs = n_same_pairs + n_diff_pairs
    all_pair_meta = []

    # determine split for pairs (pair_id will be index in this list)
    train_idx, val_idx, test_idx = _split_indices(total_pairs, splits=splits, seed=seed)
    pair_id_to_split = {}
    for i in train_idx: pair_id_to_split[i] = "train"
    for i in val_idx:   pair_id_to_split[i] = "val"
    for i in test_idx:  pair_id_to_split[i] = "test"

    # queue labels in order: first same, then diff
    labels = (["same"] * n_same_pairs) + (["diff"] * n_diff_pairs)

    for pair_id, label in enumerate(labels):
        split = pair_id_to_split[pair_id]
        # choose a bag: 3 distinct shapes + 3 distinct colors
        bag_shapes, bag_colors = _random_bag(shapes, list(colors.keys()))

        # image 0 pairings
        img0_colors_perm = bag_colors[:]
        random.shuffle(img0_colors_perm)
        img0_pairings = list(zip(bag_shapes, img0_colors_perm))

        # positions for both images (randomized per image)
        pos0 = _non_overlapping_positions(n=3, r=radius)
        pos1 = _non_overlapping_positions(n=3, r=radius)

        if label == "same":
            img1_pairings = img0_pairings[:]  # same combos
        else:
            img1_colors_perm = _diff_pair_repair(bag_shapes, img0_colors_perm)
            img1_pairings = list(zip(bag_shapes, img1_colors_perm))

        # build object dicts
        img0_objs = _compose_objects(img0_pairings, pos0, colors)
        img1_objs = _compose_objects(img1_pairings, pos1, colors)

        # render
        im0 = _render_image(img0_objs, radius=radius)
        im1 = _render_image(img1_objs, radius=radius)

        # filenames
        f0 = f"pair_{pair_id:05d}__{label}__img_0.png"
        f1 = f"pair_{pair_id:05d}__{label}__img_1.png"
        (root / split / f0).parent.mkdir(parents=True, exist_ok=True)
        im0.save(root / split / f0)
        im1.save(root / split / f1)

        # record metadata for both images
        for img_idx, objs in [(0, img0_objs), (1, img1_objs)]:
            all_pair_meta.append({
                "pair_id": pair_id,
                "label": label,         # "same" or "diff"
                "split": split,         # "train" | "val" | "test"
                "image_idx": img_idx,   # 0 or 1
                "file": str((Path(split) / (f0 if img_idx == 0 else f1)).as_posix()),
                "objects": [
                    {"shape": o["shape"], "color": o["color"], "xy": o["xy"]}
                    for o in objs
                ],
                "bag_shapes": bag_shapes,
                "bag_colors": bag_colors
            })

    with open(root / "metadata.jsonl", "w") as f:
        for row in all_pair_meta:
            f.write(json.dumps(row) + "\n")

    print(f"Done. Pairs: {total_pairs} ({n_same_pairs} same, {n_diff_pairs} diff). Output: {root.resolve()}")

# ---------- Example run ----------
if __name__ == "__main__":
    generate_dataset(
        out_dir="bag3_dataset",
        shapes=DEFAULT_SHAPES,
        colors=DEFAULT_COLORS,
        n_same_pairs=400,
        n_diff_pairs=400,
        radius=RADIUS,
        splits=(0.6, 0.2, 0.2),
        seed=1
    )
