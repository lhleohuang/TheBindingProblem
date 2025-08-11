import json, random, math
from pathlib import Path
from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle", "star", "pentagon"]
COLORS = {
    "red":    (220,  50,  47),
    "blue":   ( 38, 139, 210),
    "green":  (133, 153,  0),
    "yellow": (181, 137,   0),
    "purple": (108, 113, 196),
}

def gen_positions(w=224, h=224, r=25):
    pad = r
    while True:
        x1 = random.randint(pad, w - pad - 1)
        y1 = random.randint(pad, h - pad - 1)
        x2 = random.randint(pad, w - pad - 1)
        y2 = random.randint(pad, h - pad - 1)
        if (x1-x2)**2 + (y1-y2)**2 >= (2*r)**2:
            return (x1,y1),(x2,y2)

def draw_shape(draw, shape, center, r, fill):
    x,y = center
    if shape == "circle":
        draw.ellipse([x-r,y-r,x+r,y+r], fill=fill)
    elif shape == "square":
        draw.rectangle([x-r,y-r,x+r,y+r], fill=fill)
    elif shape == "triangle":
        pts = [
            (x, y - 2 * r / math.sqrt(3)),      # top vertex
            (x - r, y + r / math.sqrt(3)),      # bottom-left vertex
            (x + r, y + r / math.sqrt(3)),      # bottom-right vertex
        ]
        draw.polygon(pts, fill=fill)
    elif shape == "pentagon":
        pts = []
        for k in range(5):
            a = -math.pi/2 + 2*math.pi*k/5
            pts.append((x + r*math.cos(a), y + r*math.sin(a)))
        draw.polygon(pts, fill=fill)
    elif shape == "star":
        pts = []
        for k in range(10):
            a = -math.pi/2 + 2*math.pi*k/10
            rr = r if k%2==0 else r*0.45
            pts.append((x + rr*math.cos(a), y + rr*math.sin(a)))
        draw.polygon(pts, fill=fill)

def render_image(s1,c1,s2,c2, pos1,pos2, size=(224,224), r=25, bg=(255,255,255)):
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    draw_shape(d, s1, pos1, r, COLORS[c1])
    draw_shape(d, s2, pos2, r, COLORS[c2])
    return img

def all_ordered_pairs():
    out = []
    for s1 in SHAPES:
        for c1 in COLORS.keys():
            for s2 in SHAPES:
                if s2 == s1: continue
                for c2 in COLORS.keys():
                    if c2 == c1: continue
                    out.append((s1,c1,s2,c2))
    return out  # 25*16 = 400

def make_dataset(root="bag_pairs_v1", identical_ratio=0.30, seed=42, r=25):
    random.seed(seed)
    root = Path(root)
    same_dir = root/"same"
    diff_dir = root/"different"
    for p in [same_dir, diff_dir]:
        p.mkdir(parents=True, exist_ok=True)

    combos = all_ordered_pairs()  # 400 unique ordered (s1,c1,s2,c2)
    n_same = len(combos)
    n_identical = int(round(identical_ratio * n_same))
    idxs = list(range(n_same))
    random.shuffle(idxs)
    identical_idx = set(idxs[:n_identical])

    meta_same = []
    meta_diff = []

    for pair_id, (s1,c1,s2,c2) in enumerate(combos):
        pos1, pos2 = gen_positions(r=r)
        img0 = render_image(s1,c1,s2,c2,pos1,pos2, r=r)
        if pair_id in identical_idx:
            img1 = render_image(s1,c1,s2,c2,pos1,pos2, r=r)
            same_type = "identical"
        else:
            img1 = render_image(s1,c1,s2,c2,pos2,pos1, r=r)
            same_type = "swapped"

        fn0 = f"{s1}_{c1}__{s2}_{c2}__{pair_id}__0.png"
        fn1 = f"{s1}_{c1}__{s2}_{c2}__{pair_id}__1.png"
        img0.save(same_dir/fn0)
        img1.save(same_dir/fn1)

        meta_same.append({
            "pair_id": pair_id,
            "label": "same",
            "same_type": same_type,
            "image0": str((same_dir/fn0).as_posix()),
            "image1": str((same_dir/fn1).as_posix()),
            "s1": s1, "c1": c1, "s2": s2, "c2": c2,
            "pos1": {"x": pos1[0], "y": pos1[1]},
            "pos2": {"x": pos2[0], "y": pos2[1]},
        })

    for pair_id, (s1,c1,s2,c2) in enumerate(combos):
        pos1, pos2 = gen_positions(r=r)
        img0 = render_image(s1,c1,s2,c2,pos1,pos2, r=r)
        img1 = render_image(s1,c2,s2,c1,pos1,pos2, r=r)

        fn0 = f"{s1}_{c1}__{s2}_{c2}__{pair_id}__0.png"
        fn1 = f"{s1}_{c2}__{s2}_{c1}__{pair_id}__1.png"
        img0.save(diff_dir/fn0)
        img1.save(diff_dir/fn1)

        meta_diff.append({
            "pair_id": pair_id,
            "label": "different",
            "image0": str((diff_dir/fn0).as_posix()),
            "image1": str((diff_dir/fn1).as_posix()),
            "s1": s1, "c1": c1, "s2": s2, "c2": c2,
            "pos1": {"x": pos1[0], "y": pos1[1]},
            "pos2": {"x": pos2[0], "y": pos2[1]},
        })

    with open(root/"same_metadata.jsonl","w") as f:
        for row in meta_same:
            f.write(json.dumps(row)+"\n")
    with open(root/"different_metadata.jsonl","w") as f:
        for row in meta_diff:
            f.write(json.dumps(row)+"\n")

if __name__ == "__main__":
    make_dataset(
        root="bag_dataset_location_stable_1", # change to output folder name (convention is stable_0 or stable_1)
        identical_ratio=1, # stable_0 for 0, stable_1 for 1.0
        seed=42,
        r=25,
    )
