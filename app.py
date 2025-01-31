import gradio as gr
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import LTXPipeline, LTXImageToVideoPipeline, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig, T5EncoderModel
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
import moviepy.editor as mp

# Initialize pipelines at startup
print("Initializing pipelines...")

def init_pipelines():
    print("Loading text encoder...")
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    text_encoder_8bit = T5EncoderModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="text_encoder",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )

    print("Loading transformer...")
    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    transformer_8bit = LTXVideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )

    print("Creating text-to-video pipeline...")
    text_pipeline = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        text_encoder=text_encoder_8bit,
        transformer=transformer_8bit,
        torch_dtype=torch.float16,
        device_map="balanced",
    )

    print("Creating image-to-video pipeline...")
    image_pipeline = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        text_encoder=text_encoder_8bit,
        transformer=transformer_8bit,
        torch_dtype=torch.float16,
        device_map="balanced",
    )
    
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

def generate_video_from_image(image, prompt, num_inference_steps, guidance_scale, num_frames, resolution):
    print(f"Starting image-to-video generation with params: {resolution}, {num_frames} frames")
    width, height = resolution.split('x')
    width, height = int(width), int(height)
    
    try:
        print("Processing input image...")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((width, height))
        
        print("Generating video frames...")
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
        print(f"Error in image-to-video generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

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

def resize_video(video_path, target_resolution):
    """
    Resize a video to match the target resolution
    Returns path to the resized video
    """
    width, height = map(int, target_resolution.split('x'))
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            video = mp.VideoFileClip(video_path)
            resized_video = video.resize(width=width, height=height)
            resized_video.write_videofile(tmp_file.name, codec="libx264")
            video.close()
            resized_video.close()
            return tmp_file.name
    except Exception as e:
        print(f"Error resizing video: {str(e)}")
        raise

def extend_video(video_path, prompt, num_inference_steps, guidance_scale, num_frames, resolution):
    print(f"Starting video extension with params: {resolution}, {num_frames} frames")
    
    try:
        # First, resize the input video if needed
        print("Checking and resizing input video...")
        cap = cv2.VideoCapture(video_path)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        target_width, target_height = map(int, resolution.split('x'))
        if input_width != target_width or input_height != target_height:
            print(f"Resizing video from {input_width}x{input_height} to {resolution}")
            video_path = resize_video(video_path, resolution)
        
        print("Extracting last frame from video...")
        last_frame = extract_last_frame(video_path)
        
        print("Generating extension from last frame...")
        extension_path = generate_video_from_image(
            last_frame, prompt, num_inference_steps, guidance_scale, num_frames, resolution
        )
        
        if extension_path.startswith("Error"):
            return extension_path
            
        print("Concatenating videos...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            video1 = mp.VideoFileClip(video_path)
            video2 = mp.VideoFileClip(extension_path)
            final_video = mp.concatenate_videoclips([video1, video2])
            final_video.write_videofile(tmp_file.name, codec="libx264")
            
            # Clean up temporary files
            video1.close()
            video2.close()
            final_video.close()
            if os.path.exists(extension_path):
                os.unlink(extension_path)
            # Clean up resized video if it was created
            if input_width != target_width or input_height != target_height:
                os.unlink(video_path)
                
            return tmp_file.name
            
    except Exception as e:
        print(f"Error extending video: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LTX Video Generation")
    
    with gr.Tab("Text to Video"):
        text_input = gr.Textbox(
            label="Enter your prompt",
            placeholder="A detailed description of the video you want to generate..."
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
    
    with gr.Tab("Image to Video"):
        image_input = gr.Image(type="pil", label="Upload an image")
        image_text_input = gr.Textbox(
            label="Enter your prompt",
            placeholder="Describe how you want the image to animate..."
        )
        
        with gr.Row():
            image_num_inference_steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=30,
                step=1,
                label="Num Inference Steps"
            )
            image_guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.1,
                label="Guidance Scale"
            )
            image_num_frames = gr.Slider(
                minimum=16,
                maximum=128,
                value=64,
                step=16,
                label="Number of Frames"
            )
            
        image_resolution = gr.Radio(
            ["576x320", "768x432", "1024x576"],
            label="Resolution (width x height)",
            value="576x320"
        )
        
        image_generate_button = gr.Button("Generate Video")
        image_output = gr.Video(label="Generated Video")
        
        image_generate_button.click(
            generate_video_from_image,
            inputs=[
                image_input,
                image_text_input,
                image_num_inference_steps,
                image_guidance_scale,
                image_num_frames,
                image_resolution
            ],
            outputs=image_output
        )
    
    with gr.Tab("Extend Video"):
        extend_video_input = gr.Video(label="Upload a video to extend")
        extend_text_input = gr.Textbox(
            label="Enter your prompt for the extension",
            placeholder="Describe how you want the video to continue..."
        )
        
        with gr.Row():
            extend_num_inference_steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=30,
                step=1,
                label="Num Inference Steps"
            )
            extend_guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.1,
                label="Guidance Scale"
            )
            extend_num_frames = gr.Slider(
                minimum=16,
                maximum=128,
                value=64,
                step=16,
                label="Number of Frames"
            )
            
        extend_resolution = gr.Radio(
            ["576x320", "768x432", "1024x576"],
            label="Resolution (width x height)",
            value="576x320"
        )
        
        extend_button = gr.Button("Extend Video")
        extend_output = gr.Video(label="Extended Video")
        
        extend_button.click(
            extend_video,
            inputs=[
                extend_video_input,
                extend_text_input,
                extend_num_inference_steps,
                extend_guidance_scale,
                extend_num_frames,
                extend_resolution
            ],
            outputs=extend_output
        )

if __name__ == "__main__":
    # Launch with a larger queue size for video generation
    demo.queue(max_size=5)
    demo.launch()
    gr.Markdown("# LTX Video Generation")
    
    with gr.Tab("Text to Video"):
        text_input = gr.Textbox(
            label="Enter your prompt",
            placeholder="A detailed description of the video you want to generate..."
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
    
    with gr.Tab("Image to Video"):
        image_input = gr.Image(type="pil", label="Upload an image")
        image_text_input = gr.Textbox(
            label="Enter your prompt",
            placeholder="Describe how you want the image to animate..."
        )
        
        with gr.Row():
            image_num_inference_steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=30,
                step=1,
                label="Num Inference Steps"
            )
            image_guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.1,
                label="Guidance Scale"
            )
            image_num_frames = gr.Slider(
                minimum=16,
                maximum=128,
                value=64,
                step=16,
                label="Number of Frames"
            )
            
        image_resolution = gr.Radio(
            ["576x320", "768x432", "1024x576"],
            label="Resolution (width x height)",
            value="576x320"
        )
        
        image_generate_button = gr.Button("Generate Video")
        image_output = gr.Video(label="Generated Video")
        
        image_generate_button.click(
            generate_video_from_image,
            inputs=[
                image_input,
                image_text_input,
                image_num_inference_steps,
                image_guidance_scale,
                image_num_frames,
                image_resolution
            ],
            outputs=image_output
        )

if __name__ == "__main__":
    demo.launch()
