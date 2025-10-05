# clipseg_finetune_pipeline.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# --------------------------
# 1. Dataset
# --------------------------
class TextPromptSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, prompts, transform=None):
        """
        image_dir: folder with images
        mask_dir: folder with masks (binary 0/255)
        prompts: list of text prompts (str)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.prompts = prompts
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        assert len(self.images) == len(self.masks), "Image and mask counts mismatch"

    def __len__(self):
        return len(self.images) * len(self.prompts)

    def __getitem__(self, idx):
        img_idx = idx // len(self.prompts)
        prompt_idx = idx % len(self.prompts)

        img_path = os.path.join(self.image_dir, self.images[img_idx])
        mask_path = os.path.join(self.mask_dir, self.masks[img_idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask) // 255  # binary 0/1
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        prompt = self.prompts[prompt_idx]

        return image, mask, prompt, self.images[img_idx]

# --------------------------
# 2. Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# --------------------------
# 3. Load Data
# --------------------------

crack_prompts = ["segment crack", "segment wall crack"]
crack_dataset = TextPromptSegDataset(
    image_dir="/scratch/sai/clipseg/data/train",
    mask_dir="/scratch/sai/clipseg/data/masks",
    prompts=crack_prompts,
    transform=transform
)


dataloader = DataLoader(crack_dataset, batch_size=100, shuffle=True)

# --------------------------
# 4. Model Setup
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.to(device)

# --------------------------
# 5. Optimizer & Loss
# --------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# --------------------------
# 6. Training Loop
# --------------------------
epochs = 1
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, masks, prompts, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        masks = masks.to(device)

        # Prepare inputs
        images_pil = [TF.to_pil_image(img.cpu()) for img in images]

        inputs = processor(
            text=list(prompts),            
            images=images_pil,
            padding=True,                  
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)


        outputs = model(**inputs)
        logits = outputs.logits.squeeze(1)  # (B,H,W)
        masks = F.interpolate( masks.unsqueeze(1), size=logits.shape[-2:], mode="bilinear",    align_corners=False).squeeze(1)
        loss = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        

    print(f"Epoch {epoch+1} Loss: {running_loss/len(dataloader):.4f}")

# --------------------------
# 7. Evaluation
# --------------------------
def dice_coef(pred, target, smooth=1e-6):
    """
    Dice coefficient (F1 score for segmentation).
    pred, target: torch tensors of shape (H, W)
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou(pred, target, smooth=1e-6):
    """
    Intersection over Union (Jaccard index).
    pred, target: torch tensors of shape (H, W)
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


model.eval()
all_dice, all_iou = [], []
os.makedirs("pred_masks", exist_ok=True)

with torch.no_grad():
    for images, masks, prompts, img_names in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        # Convert to PIL for processor
        images_pil = [TF.to_pil_image(img.cpu()) for img in images]

        inputs = processor(
            text=list(prompts),
            images=images_pil,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        logits = outputs.logits.squeeze(1)
        probs = torch.sigmoid(logits)

        masks_resized = F.interpolate(
            masks.unsqueeze(1).float(),       
            size=probs.shape[-2:],           
            mode="nearest"                    
        ).squeeze(1)                         


        # Metrics + save masks
        for i in range(len(probs)):
            all_dice.append(dice_coef(probs[i], masks_resized[i]).item())
            all_iou.append(iou(probs[i], masks_resized[i]).item())

            mask_out = (probs[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
            filename = f"{img_names[i].split('.')[0]}__{prompts[i].replace(' ', '_')}.png"
            Image.fromarray(mask_out).save(os.path.join("pred_masks", filename))

print(f"Mean Dice: {np.mean(all_dice):.4f}, Mean IoU: {np.mean(all_iou):.4f}")


import os

os.makedirs("visualizations", exist_ok=True)

def visualize(image, gt_mask, pred_mask, save_path):
    """
    Saves a side-by-side visualization of the original image, GT mask, and predicted mask.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.permute(1,2,0).cpu())
    axes[0].set_title("Original")
    axes[1].imshow(gt_mask.cpu(), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_mask.cpu(), cmap="gray")
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)  

# --------------------------
# Example visualization loop
# --------------------------
for images, masks, prompts, img_names in dataloader:
    model.eval()

    # Convert tensors to PIL for processor
    images_pil = [TF.to_pil_image(img.cpu()) for img in images]

    inputs = processor(
        text=list(prompts),
        images=images_pil,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits.squeeze(1))

    for i in range(len(images)):
        # Resize GT mask to match prediction size
        gt_resized = F.interpolate(
            masks[i].unsqueeze(0).unsqueeze(0).float(),
            size=probs.shape[-2:],
            mode="nearest"
        ).squeeze()

        save_path = os.path.join(
            "visualizations",
            f"{img_names[i].split('.')[0]}__{prompts[i].replace(' ', '_')}.png"
        )
        visualize(images[i], gt_resized, (probs[i] > 0.5).float(), save_path)
    
    break  
