import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import random
import os
from PIL import Image, ImageTk
from text_to_image_generator import TextToImageGenerator

class TextToImageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text-to-Image Generator")
        self.root.geometry("900x700")
        
        # Initialize the generator in the background
        self.generator = None
        self.current_image = None
        self.status_var = tk.StringVar(value="Loading model... Please wait")
        
        self.setup_ui()
        
        # Start initialization in background
        self.init_thread = threading.Thread(target=self.initialize_generator)
        self.init_thread.daemon = True
        self.init_thread.start()
    
    def initialize_generator(self):
        """Initialize the text-to-image generator in a background thread."""
        try:
            self.generator = TextToImageGenerator(safety_checker=False)
            self.status_var.set("Ready to generate images")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Left panel (inputs)
        left_panel = ttk.Frame(main_frame, padding=5, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel (image display)
        right_panel = ttk.Frame(main_frame, padding=5)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Prompt input
        ttk.Label(left_panel, text="Prompt:").pack(anchor=tk.W, pady=(0, 5))
        self.prompt_text = scrolledtext.ScrolledText(left_panel, height=5, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.X, pady=(0, 10))
        self.prompt_text.insert(tk.END, "A beautiful landscape with [mountains:blue] and [trees:green]")
        
        # Negative prompt
        ttk.Label(left_panel, text="Negative Prompt:").pack(anchor=tk.W, pady=(0, 5))
        self.negative_prompt_text = scrolledtext.ScrolledText(left_panel, height=2, wrap=tk.WORD)
        self.negative_prompt_text.pack(fill=tk.X, pady=(0, 10))
        self.negative_prompt_text.insert(tk.END, "blurry, bad quality, distorted")
        
        # Parameters frame
        params_frame = ttk.LabelFrame(left_panel, text="Generation Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=10)
        
        # Width and height
        size_frame = ttk.Frame(params_frame)
        size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(size_frame, text="Width:").pack(side=tk.LEFT)
        self.width_var = tk.IntVar(value=512)
        width_combo = ttk.Combobox(size_frame, textvariable=self.width_var, width=5)
        width_combo['values'] = (256, 384, 512, 576, 640, 768)
        width_combo.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(size_frame, text="Height:").pack(side=tk.LEFT)
        self.height_var = tk.IntVar(value=512)
        height_combo = ttk.Combobox(size_frame, textvariable=self.height_var, width=5)
        height_combo['values'] = (256, 384, 512, 576, 640, 768)
        height_combo.pack(side=tk.LEFT, padx=5)
        
        # Steps and guidance
        steps_frame = ttk.Frame(params_frame)
        steps_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(steps_frame, text="Steps:").pack(side=tk.LEFT)
        self.steps_var = tk.IntVar(value=30)
        steps_spin = ttk.Spinbox(steps_frame, from_=10, to=100, textvariable=self.steps_var, width=5)
        steps_spin.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(steps_frame, text="Guidance:").pack(side=tk.LEFT)
        self.guidance_var = tk.DoubleVar(value=7.5)
        guidance_spin = ttk.Spinbox(steps_frame, from_=1.0, to=20.0, increment=0.5, textvariable=self.guidance_var, width=5)
        guidance_spin.pack(side=tk.LEFT, padx=5)
        
        # Seed
        seed_frame = ttk.Frame(params_frame)
        seed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(seed_frame, text="Seed:").pack(side=tk.LEFT)
        self.seed_var = tk.IntVar(value=-1)
        seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var, width=10)
        seed_entry.pack(side=tk.LEFT, padx=5)
        
        # Random seed button
        random_seed_btn = ttk.Button(seed_frame, text="Random", command=self.set_random_seed)
        random_seed_btn.pack(side=tk.LEFT, padx=5)
        
        # Watermark checkbox
        self.watermark_var = tk.BooleanVar(value=False)
        watermark_check = ttk.Checkbutton(params_frame, text="Apply Watermark", variable=self.watermark_var)
        watermark_check.pack(anchor=tk.W, pady=5)
        
        # Model selection
        model_frame = ttk.Frame(left_panel)
        model_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="runwayml/stable-diffusion-v1-5")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=40)
        model_combo['values'] = (
            "runwayml/stable-diffusion-v1-5",
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-2-1",
            "dreamlike-art/dreamlike-diffusion-1.0"
        )
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        generate_btn = ttk.Button(buttons_frame, text="Generate Image", command=self.generate_image)
        generate_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = ttk.Button(buttons_frame, text="Save Image", command=self.save_image)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Image display
        self.image_frame = ttk.LabelFrame(right_panel, text="Generated Image", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Help text
        help_text = "Use [element:color] syntax in your prompt to specify colors.\n"
        help_text += "Example: 'A castle with [walls:red] and [roof:blue]'"
        help_label = ttk.Label(left_panel, text=help_text, foreground="gray")
        help_label.pack(anchor=tk.W, pady=10)
    
    def set_random_seed(self):
        """Set a random seed value."""
        self.seed_var.set(random.randint(1, 2147483647))
    
    def generate_image(self):
        """Generate an image based on the current settings."""
        if self.generator is None:
            self.status_var.set("Model is still loading. Please wait...")
            return
        
        # Get parameters from UI
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        negative_prompt = self.negative_prompt_text.get("1.0", tk.END).strip()
        width = self.width_var.get()
        height = self.height_var.get()
        steps = self.steps_var.get()
        guidance = self.guidance_var.get()
        seed = self.seed_var.get() if self.seed_var.get() >= 0 else None
        apply_watermark = self.watermark_var.get()
        
        # Disable UI during generation
        self.status_var.set("Generating image...")
        self.root.update()
        
        # Generate in a separate thread to keep UI responsive
        def generate_thread():
            try:
                # Generate the image
                image = self.generator.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed,
                    apply_watermark=apply_watermark
                )
                
                # Store the image and update display
                self.current_image = image
                self.display_image(image)
                self.status_var.set("Image generated successfully")
            except Exception as e:
                self.status_var.set(f"Error generating image: {str(e)}")
        
        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()
    
    def display_image(self, image):
        """Display an image in the UI."""
        # Resize image to fit in the display area while maintaining aspect ratio
        display_width = self.image_frame.winfo_width() - 20
        display_height = self.image_frame.winfo_height() - 20
        
        if display_width <= 1 or display_height <= 1:
            # Frame not properly sized yet, use default size
            display_width = 400
            display_height = 400
        
        # Calculate scaling factor
        width_ratio = display_width / image.width
        height_ratio = display_height / image.height
        scale_factor = min(width_ratio, height_ratio)
        
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        # Resize image for display
        display_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage for Tkinter
        tk_image = ImageTk.PhotoImage(display_image)
        
        # Update label
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image  # Keep a reference to prevent garbage collection
    
    def save_image(self):
        """Save the current image to a file."""
        if self.current_image is None:
            self.status_var.set("No image to save")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            self.current_image.save(file_path)
            self.status_var.set(f"Image saved to {file_path}")
        except Exception as e:
            self.status_var.set(f"Error saving image: {str(e)}")

def main():
    root = tk.Tk()
    app = TextToImageGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
