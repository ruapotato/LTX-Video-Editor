import gradio as gr
import torch
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from transformers import BitsAndBytesConfig
from diffusers.utils import export_to_video
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
import moviepy.editor as mp

# Initialize pipelines at startup
print("Initializing pipelines...")

def init_pipelines():
    print("Setting up quantization configs...")
    # Using the recommended quantization settings from the docs
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        device_map={"": torch.device("cuda:0")}
    )
    
    print("Creating text-to-video pipeline...")
    text_pipeline = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.float16,  # Changed to float16 for compatibility with 8-bit quantization
        quantization_config=quant_config
    )
    text_pipeline.enable_model_cpu_offload()
    
    print("Creating image-to-video pipeline...")
    image_pipeline = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.float16,  # Changed to float16 for compatibility with 8-bit quantization
        quantization_config=quant_config
    )
    image_pipeline.enable_model_cpu_offload()
    
    return text_pipeline, image_pipeline

# Global pipeline instances
TEXT_PIPELINE, IMAGE_PIPELINE = init_pipelines()
print("Pipelines initialized successfully!")

def generate_video_from_text(prompt, num_inference_steps, guidance_scale, num_frames, resolution):
    print(f"Starting text-to-video generation with params: {resolution}, {num_frames} frames")
    width, height = resolution.split('x')
    width, height = int(width), int(height)
    
    try:
        print("Generating video frames...")
        with torch.inference_mode():  # Added for better memory efficiency
            output = TEXT_PIPELINE(
                prompt=prompt,
                negative_prompt="worst quality, low quality, blurry, distorted",
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
            )
        
        print("Exporting to video file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            export_to_video(output.frames[0], tmp_file.name, fps=24)
            print(f"Video saved to {tmp_file.name}")
            return tmp_file.name
            
    except Exception as e:
        print(f"Error in text-to-video generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"
    finally:
        # Force CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def extract_last_frame(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Unable to read the last frame")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = Image.fromarray(frame_rgb)
        
        cap.release()
        return last_frame
        
    except Exception as e:
        print(f"Error extracting last frame: {str(e)}")
        raise

def generate_video_from_source(source, prompt, num_inference_steps, guidance_scale, num_frames, resolution):
    print(f"Starting source-to-video generation with params: {resolution}, {num_frames} frames")
    width, height = resolution.split('x')
    width, height = int(width), int(height)
    
    try:
        print("Processing input source...")
        if isinstance(source, str) and os.path.exists(source):  # Video input
            image = extract_last_frame(source)
        elif isinstance(source, np.ndarray):  # Image input as numpy array
            image = Image.fromarray(source)
        else:  # Direct PIL Image input
            image = source
            
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((width, height))
        
        print("Generating video frames...")
        with torch.inference_mode():  # Added for better memory efficiency
            output = IMAGE_PIPELINE(
                image=image,
                prompt=prompt,
                negative_prompt="worst quality, low quality, blurry, distorted",
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
            )
        
        print("Exporting to video file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            export_to_video(output.frames[0], tmp_file.name, fps=24)
            print(f"Video saved to {tmp_file.name}")
            return tmp_file.name
            
    except Exception as e:
        print(f"Error in source-to-video generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"
    finally:
        # Force CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LTX Video Generation")
    
    with gr.Tab("Text to Video"):
        text_input = gr.Textbox(
            label="Enter your prompt",
            placeholder="A detailed description of the video you want to generate...",
            lines=3
        )
        
        with gr.Row():
            text_num_inference_steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=30,
                step=1,
                label="Num Inference Steps"
            )
            text_guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.1,
                label="Guidance Scale"
            )
            text_num_frames = gr.Slider(
                minimum=16,
                maximum=128,
                value=64,
                step=16,
                label="Number of Frames"
            )
            
        text_resolution = gr.Radio(
            ["576x320", "768x432", "1024x576"],
            label="Resolution (width x height)",
            value="576x320"
        )
        
        text_generate_button = gr.Button("Generate Video")
        text_output = gr.Video(label="Generated Video")
        
        text_generate_button.click(
            generate_video_from_text,
            inputs=[
                text_input,
                text_num_inference_steps,
                text_guidance_scale,
                text_num_frames,
                text_resolution
            ],
            outputs=text_output
        )
    
    with gr.Tab("Image/Video to Video"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload an image")
            video_input = gr.Video(label="Or upload a video (last frame will be used)")
        
        source_text_input = gr.Textbox(
            label="Enter your prompt",
            placeholder="Describe how you want the image/video to animate...",
            lines=3
        )
        
        with gr.Row():
            source_num_inference_steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=30,
                step=1,
                label="Num Inference Steps"
            )
            source_guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.1,
                label="Guidance Scale"
            )
            source_num_frames = gr.Slider(
                minimum=16,
                maximum=128,
                value=64,
                step=16,
                label="Number of Frames"
            )
            
        source_resolution = gr.Radio(
            ["576x320", "768x432", "1024x576"],
            label="Resolution (width x height)",
            value="576x320"
        )
        
        source_generate_button = gr.Button("Generate Video")
        source_output = gr.Video(label="Generated Video")
        
        def handle_source_input(image, video, prompt, steps, guidance, frames, resolution):
            source = video if video is not None else image
            if source is None:
                return "Error: Please provide either an image or video input"
            return generate_video_from_source(source, prompt, steps, guidance, frames, resolution)
        
        source_generate_button.click(
            handle_source_input,
            inputs=[
                image_input,
                video_input,
                source_text_input,
                source_num_inference_steps,
                source_guidance_scale,
                source_num_frames,
                source_resolution
            ],
            outputs=source_output
        )

if __name__ == "__main__":
    # Launch with a larger queue size for video generation
    demo.queue(max_size=5)
    demo.launch()
