import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

class TextToImageGenerator:
    """A text-to-image generator with color customization."""
    
    def __init__(
        self, 
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        safety_checker: bool = False,  # Safety checker disabled by default
        output_dir: str = "generated_images"
    ):
        """
        Initialize the text-to-image generator.
        
        Args:
            model_id: The Hugging Face model ID to use
            device: The device to run the model on ('cuda' or 'cpu')
            safety_checker: Whether to use the safety checker (disabled by default)
            output_dir: Directory to save generated images
        """
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model
        print(f"Loading model {model_id} on {device}...")
        
        # Load pipeline with safety_checker disabled
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None  # Disable safety checker
        )
        
        # Use DPM-Solver++ for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)
        
        # Enable memory efficient attention if using CUDA
        if device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print("Model loaded successfully!")
    
    def parse_color_instructions(self, prompt: str) -> Tuple[str, Dict[str, str]]:
        """
        Parse color instructions from the prompt.
        
        Args:
            prompt: The text prompt with color instructions
            
        Returns:
            Tuple of (cleaned_prompt, color_dict)
        """
        # Pattern to match color instructions like [sky:blue] or [grass:green]
        pattern = r'\[([^:]+):([^\]]+)\]'
        
        # Find all color instructions
        color_dict = {}
        matches = re.findall(pattern, prompt)
        
        for element, color in matches:
            color_dict[element.strip()] = color.strip()
        
        # Remove color instructions from the prompt
        cleaned_prompt = re.sub(pattern, r'\1', prompt)
        
        return cleaned_prompt, color_dict
    
    def enhance_prompt_with_colors(self, prompt: str, color_dict: Dict[str, str]) -> str:
        """
        Enhance the prompt with color information.
        
        Args:
            prompt: The original prompt
            color_dict: Dictionary mapping elements to colors
            
        Returns:
            Enhanced prompt with color information
        """
        enhanced_prompt = prompt
        
        for element, color in color_dict.items():
            # Add color information to the prompt in a way the model understands
            enhanced_prompt += f", {element} is {color} colored"
        
        return enhanced_prompt
    
    def generate_image(
        self, 
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        apply_watermark: bool = False,
        filename: Optional[str] = None
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: The text prompt
            negative_prompt: Things to avoid in the image
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            apply_watermark: Whether to apply a watermark
            filename: Filename to save the image (without extension)
            
        Returns:
            Generated PIL Image
        """
        # Parse color instructions
        cleaned_prompt, color_dict = self.parse_color_instructions(prompt)
        
        # Enhance prompt with color information
        if color_dict:
            enhanced_prompt = self.enhance_prompt_with_colors(cleaned_prompt, color_dict)
            print(f"Enhanced prompt: {enhanced_prompt}")
        else:
            enhanced_prompt = cleaned_prompt
        
        # Set seed for reproducibility if provided
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate the image
        print(f"Generating image with prompt: {enhanced_prompt}")
        with torch.no_grad():
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        # Apply watermark if requested
        if apply_watermark:
            image = self._apply_watermark(image)
        
        # Save the image if filename is provided
        if filename:
            save_path = os.path.join(self.output_dir, f"{filename}.png")
            image.save(save_path)
            print(f"Image saved to {save_path}")
        
        return image
    
    def _apply_watermark(self, image: Image.Image) -> Image.Image:
        """Apply a watermark to the image."""
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        
        watermark = "AI Generated"
        draw.text((10, image.height - 25), watermark, fill=(255, 255, 255, 128), font=font)
        
        return image
    
    def batch_generate(
        self, 
        prompts: List[str],
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate multiple images from a list of prompts.
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional arguments for generate_image
            
        Returns:
            List of generated PIL Images
        """
        images = []
        for i, prompt in enumerate(prompts):
            filename = kwargs.pop('filename', None)
            if filename is None:
                filename = f"image_{i+1}"
            
            image = self.generate_image(prompt, filename=filename, **kwargs)
            images.append(image)
        
        return images

def main():
    parser = argparse.ArgumentParser(description="Text-to-Image Generator with Color Customization")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="generated_image", help="Output filename (without extension)")
    parser.add_argument("--watermark", action="store_true", help="Apply watermark")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Model ID to use")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    # Determine device
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    
    # Initialize generator
    generator = TextToImageGenerator(model_id=args.model, device=device)
    
    # Generate image
    generator.generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        apply_watermark=args.watermark,
        filename=args.output
    )

if __name__ == "__main__":
    main()