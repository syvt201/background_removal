import os
import cv2
import numpy as np

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
    
