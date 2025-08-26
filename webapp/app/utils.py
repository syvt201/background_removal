import os
import cv2
import numpy as np
from torchvision import transforms
import torch
import segmentation_models_pytorch as smp
import re

def resize_and_pad(img, size=512, pad_color=0):
    h, w = img.shape[:2]
    
    scale = size / max(h, w)
    new_h = int(scale * h)
    new_w = int(scale * w)
    
    # Resize with INTER_AREA for Downsampling, INTER_LINEAR for Upsampling
    if scale < 1: 
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
        
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # Padding
    pad_w = size - new_w
    pad_h = size - new_h
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=pad_color)
    
    return img_padded, top, bottom, left, right

def unpad_and_resize(image, org_w, org_h, top, bottom, left, right):
    cropped_img = image[top:image.shape[0] - bottom,
                                left:image.shape[1] - right]
    
    h, w = cropped_img.shape[:2]
    
    scale = h / max(org_w, org_h)
    
    # Resize with INTER_AREA for Downsampling, INTER_LINEAR for Upsampling
    if scale < 1: # Upsample
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_AREA
        
    org_img = cv2.resize(cropped_img, (org_w, org_h), interpolation=interpolation)
    
    return org_img
def save_image(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    success = cv2.imwrite(save_path, image)
    
    if not success:
        print(f"[Error] Failed to save image to {save_path}")
    return success

def apply_mask(image, mask):
    mask = (mask >= 1).astype(np.uint8)

    if image.ndim == 3 and mask.ndim == 2:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
    masked_image = image * mask
    return masked_image

def get_tensor_image(image):
    transform =transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def get_mask(logit):
    mask = torch.sigmoid(logit).squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

def change_bg_color(image, mask, bg_color):
    if image.shape[:2] != mask.shape:
        raise ValueError("Mask and image must have the same height and width.")

    # Create a background layer filled with the chosen background color
    bg_layer = np.zeros_like(image, dtype=np.uint8)
    bg_layer[:, :] = bg_color

    # Expand the mask to 3 channels (H, W, 3)
    mask_3d = np.stack([mask] * 3, axis=-1)

    # Combine foreground from the original image with the new background
    result = np.where(mask_3d > 0, image, bg_layer)

    return result


def get_model(model_name, encoder_name='resnet34', encoder_weights='imagenet', num_classes=1, activation=None):

    model_name = model_name.lower()

    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation
        )

    elif model_name == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation
        )

    else:
        raise ValueError(f"Unsupported model_name '{model_name}' — choose from ['unet', 'deeplabv3']")

    return model

def load_model(checkpoint_path,device):
    model = get_model(model_name="unet", num_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def parse_color_to_hex(color):
    # Check if the input is already a hex color (e.g., #FF0000)
    if isinstance(color, str) and color.startswith('#') and len(color) == 7:
        return color.upper()  # Return hex as-is, in uppercase for consistency

    # Handle rgba() format (e.g., rgba(90.7761245727539, 76.81160076623341, 76.81160076623341, 1))
    rgba_match = re.match(r'rgba?\((\d+\.?\d*), (\d+\.?\d*), (\d+\.?\d*), ?(\d*\.?\d*)\)', color)
    if rgba_match:
        # Extract RGB values and convert to integers
        r, g, b = int(float(rgba_match.group(1))), int(float(rgba_match.group(2))), int(float(rgba_match.group(3)))
        # Convert to hex format
        return '#{:02X}{:02X}{:02X}'.format(r, g, b)
    
    raise ValueError(f"Invalid color format: {color}")

def hex_to_rgb(hex_color):
    # Remove "#" and split into (R, G, B)
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb