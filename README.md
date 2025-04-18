# Text-to-Image Generator with Color Customization

This application generates images from text descriptions with color customization using Stable Diffusion.

## Features

- Generate images from text prompts
- Customize colors of specific elements using [element:color] syntax
- Adjust image size, generation steps, and guidance
- Set random seeds for reproducibility
- Simple and intuitive GUI

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/text-to-image-generator.git
cd text-to-image-generator
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### GUI Application

Run the GUI application:
```
python text_to_image_gui.py
```

### Command Line Interface

Generate an image using the command line:
```
python text_to_image_generator.py --prompt "A beautiful landscape with [mountains:blue] and [trees:green]"
```

Additional options:
```
python text_to_image_generator.py --help
```

## Color Customization Syntax

Use the following syntax in your prompts to customize colors:
```
[element:color]
```

Examples:
- "A [house:red] with a [roof:brown] and [door:blue]"
- "A [sky:purple] with [clouds:pink] over a [field:golden]"

## Requirements

- Python 3.7+
- CUDA-capable GPU recommended for faster generation

## License

MIT
```

## How to Use This Text-to-Image Generator

1. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Run the GUI application**:
   ```
   python text_to_image_gui.py