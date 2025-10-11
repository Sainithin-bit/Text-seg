# convert_coco_to_png_masks.py
import json
import os
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

# --------------------------
# Config
# --------------------------
coco_json_path = "/scratch/sai/clipseg/data/data_2/train/_annotations.coco.json"  # COCO JSON exported from Roboflow
images_dir = "/scratch/sai/clipseg/data/data_2/train"  # folder with original images
masks_dir = "/scratch/sai/clipseg/data/data_2/masks"  # output folder
class_prompt_map = {
    "crack": "segment crack",
    "NewCracks": "segment wall crack"
}
image_size = (512, 512)  # resize masks to this

os.makedirs(masks_dir, exist_ok=True)

# --------------------------
# Load COCO JSON
# --------------------------
with open(coco_json_path) as f:
    coco = json.load(f)

# Build annotation dict
img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
annotations_by_image = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    annotations_by_image.setdefault(img_id, []).append(ann)

# Build category dict
cat_id_to_name = {1:"crack", 0:"NewCracks"}

# --------------------------
# Convert polygons to masks
# --------------------------
def polygons_to_mask(polygons, size):
    mask = Image.new("L", size, 0)
    for seg in polygons:
        xy = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
        ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)
    return mask

for img_id, file_name in tqdm(img_id_to_file.items(), desc="Converting masks"):
    anns = annotations_by_image.get(img_id, [])
    img_path = os.path.join(images_dir, file_name)
    pil_img = Image.open(img_path)
    original_size = pil_img.size  # (W,H)
    # import pdb;pdb.set_trace()
    for ann in anns:
        cat_name = cat_id_to_name[ann['category_id']]

        # if cat_name not in class_prompt_map:
        #     continue

        prompt = class_prompt_map[cat_name]
        polygons = ann['bbox']
        mask = polygons_to_mask(polygons, original_size)

        # Resize mask
        mask = mask.resize(image_size, resample=Image.NEAREST)

        # Convert to 0/255
        mask = np.array(mask) * 255
        mask_img = Image.fromarray(mask.astype(np.uint8))

        # Save mask
        mask_filename = f"{file_name}"
        mask_img.save(os.path.join(masks_dir, mask_filename))
        print(mask_filename)
