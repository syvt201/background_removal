import torch
import cv2
import numpy as np
from PIL import Image, ImageFilter
from utils import resize_and_pad, unpad_and_resize, apply_mask, get_tensor_image, get_mask

def inference_image(model, device, image=None, image_path=None, filter_size=3):
    if image_path and image is None:
        image = cv2.imread(image_path)
        
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image, top, bottom, left, right = resize_and_pad(image, size=512, pad_color=0)   
    image_tensor = get_tensor_image(processed_image)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    mask = get_mask(output)
    mask = unpad_and_resize(mask, w, h, top, bottom, left, right)
    
    # smooth edges
    pil_mask = Image.fromarray(mask)
    pil_mask = pil_mask.filter(ImageFilter.ModeFilter(size=filter_size))
    return np.array(pil_mask)
