import torch
import numpy as np
from torchvision import transforms
import argparse
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from utils.get_model import get_model
from utils.image_processing import resize_and_pad, unpad_and_resize, save_image
import time

def load_model(checkpoint_path, device):
    model = get_model(model_name="unet", num_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

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

def apply_mask(image, mask):
    mask = (mask >= 1).astype(np.uint8)

    if image.ndim == 3 and mask.ndim == 2:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
    masked_image = image * mask
    return masked_image

def infer(model, image_path, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
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
    save_mask_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0]  + ".png")
    save_image(mask, save_mask_path)
    return mask, save_mask_path

def infer_video(model, video_path, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print(f"Cannot open video {video_path}")
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0]  + ".mp4"), 
                          fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        processed_frame, top, bottom, left, right = resize_and_pad(frame, size=512, pad_color=0)   
        frame_tensor = get_tensor_image(processed_frame)
        frame_tensor = frame_tensor.to(device)
        
        with torch.no_grad():
            output = model(frame_tensor)
        
        mask = get_mask(output)
        mask = unpad_and_resize(mask, w, h, top, bottom, left, right)
        masked_frame = apply_mask(frame, mask)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint/model")
    parser.add_argument("--input_dir", type=str, default="", help="Folder of input images")
    parser.add_argument("--image_path", type=str, default="", help="Input image path")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save predicted masks")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_path, device)
    # mask, save_mask_path = infer(model, args.input_dir, args.output_dir, device)
    if args.input_dir:
        image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.png'))] 
        print(image_files)
        for fname in tqdm(image_files, desc="Running inference"):
            infer(model, os.path.join(args.input_dir, fname), args.output_dir, device)
    else:
        infer(model, args.image_path, args.output_dir, device)
    
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # checkpoint_path = "../models/unet_checkpoint.pth"
    # checkpoint_path = "../checkpoints/checkpoints_no_obj/checkpoints/checkpoint_last_epoch.pth"
    # model = load_model(checkpoint_path, device)
    # # video_path = "../video/vd.mp4"
    # video_folder = "../video"
    # output_dir = "../out"
    # for file in os.listdir(video_folder):
    #     video_path = os.path.join(video_folder, file)
        
    #     start_time = time.perf_counter()  # High-resolution timer
    #     infer_video(model, video_path, output_dir, device)
    #     end_time = time.perf_counter()
        
    #     execution_time = end_time - start_time
    #     print(f"Execution time for {video_path} : {execution_time:.6f} seconds")
    