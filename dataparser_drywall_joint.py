# convert_coco_to_png_masks.py
import json
import os
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

# --------------------------
# Config
# --------------------------
coco_json_path = "/scratch/sai/clipseg/data/data_2/_annotations.coco.json"  # COCO JSON exported from Roboflow
images_dir = "/scratch/sai/clipseg/data/data_2/train"  # folder with original images
masks_dir = "/scratch/sai/clipseg/data/data_2/masks"  # output folder
image_size = (512, 512)                                                     # Resize masks to this

# Mapping COCO class names to text prompts
class_prompt_map = {
    "Drywall-Join": "segment drywall seam",
    "drywall-join": "segment taping area"
}

os.makedirs(masks_dir, exist_ok=True)

# --------------------------
# Load COCO JSON
# --------------------------
with open(coco_json_path, "r") as f:
    coco = json.load(f)

# Map image IDs to file names
img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}

# Group annotations by image ID
annotations_by_image = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    annotations_by_image.setdefault(img_id, []).append(ann)

# Map category IDs to class names
cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

# --------------------------
# Utility: Convert bbox/polygons to mask
# --------------------------
def bbox_to_mask(bbox, size):
    """
    Converts a bounding box [x, y, width, height] into a binary mask of given size.
    """
    mask = Image.new("L", size, 0)
    x, y, w, h = bbox
    xy = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)
    return mask

# --------------------------
# Generate masks
# --------------------------
for img_id, file_name in tqdm(img_id_to_file.items(), desc="Converting masks"):
    anns = annotations_by_image.get(img_id, [])
    img_path = os.path.join(images_dir, file_name)

    try:
        pil_img = Image.open(img_path)
        original_size = pil_img.size  # (width, height)
    except Exception as e:
        print(f"Failed to open image {file_name}: {e}")
        continue

    combined_mask = Image.new("L", original_size, 0)

    for ann in anns:
        cat_name = cat_id_to_name[ann['category_id']]
        if cat_name not in class_prompt_map:
            continue

        bbox = ann['bbox']  # [x, y, width, height]
        mask = bbox_to_mask(bbox, original_size)
        # Combine multiple annotations into one mask
        combined_mask = Image.fromarray(
            np.maximum(np.array(combined_mask), np.array(mask))
        )

    # Resize to target size
    combined_mask = combined_mask.resize(image_size, resample=Image.NEAREST)

    # Convert to 0/255
    mask_array = np.array(combined_mask).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_array)

    # Save mask
    mask_filename = file_name  # or f"{file_name.split('.')[0]}_mask.png"
    mask_img.save(os.path.join(masks_dir, mask_filename))
