import os
import random
import math
import itertools
from PIL import Image, ImageDraw
import pandas as pd
from sklearn.model_selection import train_test_split

output_dir = 'bag_dataset'

# 1. Vocabulary & canvas settings
shapes = ['circle', 'triangle', 'square', 'star', 'pentagon']
colors = {
    'red':    (255,   0,   0),
    'blue':   (  0,   0, 255),
    'green':  (  0, 255,   0),
    'yellow': (255, 255,   0),
    'purple': (128,   0, 128),
}

canvas_size = 224
shape_radius = 25  # half the width/diameter of each shape

# 2. Output dirs
images_dir = output_dir
os.makedirs(images_dir, exist_ok=True)

# 3. Helpers for placement & drawing
def sample_positions():
    """Sample two centers so that shapes do not overlap."""
    while True:
        x1 = random.randint(shape_radius, canvas_size - shape_radius)
        y1 = random.randint(shape_radius, canvas_size - shape_radius)
        x2 = random.randint(shape_radius, canvas_size - shape_radius)
        y2 = random.randint(shape_radius, canvas_size - shape_radius)
        if (x1-x2)**2 + (y1-y2)**2 >= (2*shape_radius)**2:
            return (x1, y1), (x2, y2)

def regular_polygon(center, radius, n_sides):
    cx, cy = center
    return [
        (cx + radius*math.cos(2*math.pi*i/n_sides - math.pi/2),
         cy + radius*math.sin(2*math.pi*i/n_sides - math.pi/2))
        for i in range(n_sides)
    ]

def star_points(center, outer_r, inner_r, n_points=5):
    cx, cy = center
    pts = []
    for i in range(2*n_points):
        angle = math.pi/2 + i*math.pi/n_points
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r*math.cos(angle), cy + r*math.sin(angle)))
    return pts

def draw_shape(draw, shape, color, center):
    x, y = center
    if shape == 'circle':
        draw.ellipse([x-shape_radius, y-shape_radius, x+shape_radius, y+shape_radius], fill=color)
    elif shape == 'square':
        draw.rectangle([x-shape_radius, y-shape_radius, x+shape_radius, y+shape_radius], fill=color)
    elif shape == 'triangle':
        draw.polygon(regular_polygon(center, shape_radius, 3), fill=color)
    elif shape == 'pentagon':
        draw.polygon(regular_polygon(center, shape_radius, 5), fill=color)
    elif shape == 'star':
        draw.polygon(star_points(center, shape_radius, shape_radius*0.5), fill=color)
    else:
        raise ValueError(f"Unknown shape: {shape}")

# 4. Generate only assignments where both shape and color differ between objects
base_assignments = [
    (s1, c1, s2, c2)
    for s1, c1, s2, c2 in itertools.product(shapes, colors.keys(), shapes, colors.keys())
    if (s1 != s2 and c1 != c2)
]

# 5. Pre-render two images per valid assignment
for (s1, c1, s2, c2) in base_assignments:
    for idx in [0, 1]:
        img = Image.new('RGB', (canvas_size, canvas_size), 'white')
        draw = ImageDraw.Draw(img)
        pos1, pos2 = sample_positions()
        draw_shape(draw, s1, colors[c1], pos1)
        draw_shape(draw, s2, colors[c2], pos2)
        fname = f"{s1}_{c1}__{s2}_{c2}__{idx}.png"
        img.save(os.path.join(images_dir, fname))

# 6. Build paired‐image metadata
pairs = []
for (s1, c1, s2, c2) in base_assignments:
    # same‐label pair
    f1 = f"{s1}_{c1}__{s2}_{c2}__0.png"
    f2 = f"{s1}_{c1}__{s2}_{c2}__1.png"
    pairs.append((f1, f2, 'same'))

    # different‐label pair (swap colors)
    swapped = f"{s1}_{c2}__{s2}_{c1}__0.png"
    pairs.append((f1, swapped, 'different'))

# 7. Save metadata and split
df = pd.DataFrame(pairs, columns=['img1', 'img2', 'label'])
train, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

for split_name, split_df in zip(['train', 'val', 'test'], [train, val, test]):
    split_df.to_csv(os.path.join(output_dir, f"{split_name}_metadata.csv"), index=False)