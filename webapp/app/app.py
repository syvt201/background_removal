import gradio as gr
import torch
from inference import inference_image, apply_mask
from utils import load_model, change_bg_color, hex_to_rgb, parse_color_to_hex

# ==========================
# Model loading
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "../models/unet_weights.pth"

model = load_model(checkpoint_path, device)

# ==========================
# Global cache for images
# ==========================
image_cache = {"input": None, "output": None, "mask": None}

bg_colors = {
    "White": [255, 255, 255],
    "Black": [0, 0, 0],
    "Blue": [65, 12, 217],
    "Pink": [255, 170, 170],
    "Yellow": [255, 236, 141],
    "Custom": None,
}
# ==========================
# Processing function
# ==========================
def process(img, filter_size=13):
    if img is None:
        return {
            output_image: gr.update(value=None),
            colors: gr.update(elem_classes="hide"),
            error_msg: "⚠ Please upload an image before processing."
        }
    
    mask = inference_image(model, device, img, filter_size=filter_size)

    # Save to cache
    image_cache["input"] = img
    image_cache["mask"] = mask
    image_cache["output"] = apply_mask(img, mask)

    return {
        output_image: gr.update(value=image_cache["output"]),
        colors: gr.update(elem_classes=[], value="Black"),     
        error_msg: "",
        submit_btn: gr.update(elem_classes=["hide"])                             
    }

# ==========================
# Toggle background color
# ==========================
def toggle_bg_color(color):
    if color in bg_colors:
        if color == "Custom":
            return {color_picker: gr.update(elem_classes=[])}
        
        image_cache["output"] = change_bg_color(image_cache["input"], image_cache["mask"], bg_colors[color])
        return {
            output_image: gr.update(value=image_cache["output"]),
            color_picker: gr.update(elem_classes=["hide"])
        } 
        
    return gr.update()

# ==========================
# Clear image
# ==========================
def clear_img():
    return {
        output_image: gr.update(value=None),
        colors: gr.update(elem_classes=["hide"]),    
        color_picker: gr.update(elem_classes=["hide"]),
        submit_btn: gr.update(elem_classes=[])
    }

# ==========================
# Custom background color
# ==========================  
def apply_custom_bg_color(color):
    hex_color = parse_color_to_hex(color) if 'rgb' in color else color
    bg_color = hex_to_rgb(hex_color)
    image_cache["output"] = change_bg_color(image_cache["input"], image_cache["mask"], bg_color)
    return {output_image: gr.update(value=image_cache["output"])}


# ==========================
# CSS for styling
# ==========================
css = """
    #submit_btn {
        background-color: #2fc320;
        color: white;
        font-weight: bold;
    }
    #col_max_width {
        width: 100%;
        max-width: 70vw;
        margin-left: auto;
        margin-right: auto;
    }
    
    #warning_msg {
        color: red;
    }
    
    #color_picker {
        z-index: 100;
        position: fixed;
        min-height: 85px;
        max-width: 326px;
    }
    
    .hide {
        display: none !important;
        # visibility: hidden !important;
    }
    
    #blur_color_picker {
        max-width: 130px;
    }
    
    span:has(> button[aria-label="Upload file"]) {
        display: none !important;
    }
    

"""

# ==========================
# UI layout
# ==========================
with gr.Blocks(css=css, fill_height=True) as demo:
    with gr.Column(elem_id="col_max_width"):
                    
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                colors = gr.Radio(
                    bg_colors.keys(), 
                    label="Choose color", 
                    value="Black", 
                    interactive=True,
                    elem_classes=["hide"],
                )
                
            with gr.Column(scale=1):
                color_picker = gr.ColorPicker(elem_classes=["hide"], elem_id="color_picker", interactive=True, value="#000000")

        with gr.Row(equal_height=True):
            input_image = gr.Image(type="numpy", height=600, label="Image")
            output_image = gr.Image(type="numpy", height=600, label="Image", format="png")

        error_msg = gr.HTML("", elem_id="warning_msg")

        submit_btn = gr.Button(
            value="Submit", 
            # icon="../assests/submit_icon.png", 
            elem_id="submit_btn"
        )

        submit_btn.click(
            fn=process,
            inputs=input_image,
            outputs=[output_image, colors, error_msg, submit_btn]
        )

        input_image.clear(
            fn=clear_img,
            outputs=[output_image, colors, color_picker, submit_btn]
        )
        
        colors.select(
            fn=toggle_bg_color,
            inputs=colors,
            outputs=[output_image, color_picker]
        )
        
        color_picker.change(
            fn=apply_custom_bg_color,
            inputs=color_picker,
            outputs=[output_image]
        )

if __name__ == "__main__":
    demo.launch()
