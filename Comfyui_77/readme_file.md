# Alo77 - ComfyUI Custom Nodes Collection

A comprehensive collection of three powerful ComfyUI custom nodes for advanced image processing workflows.

## 📁 Folder Structure

```
ComfyUI/custom_nodes/Alo77/
├── __init__.py                 # Main initialization file
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── loadimages77/
│   ├── __init__.py
│   └── load_node.py           # LoadImages-77 node
├── canvas77/
│   ├── __init__.py
│   └── canvas_node.py         # Canvas-77 node
└── outline77/
    ├── __init__.py
    └── outline_node.py        # Outline-77 node
```

## 🚀 Installation

1. **Clone or download** this folder to your ComfyUI custom_nodes directory:
   ```
   ComfyUI/custom_nodes/Alo77/
   ```

2. **Install dependencies** (navigate to the Alo77 folder and run):
   ```bash
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI** to load the new nodes.

## 📦 Included Nodes

### 1. **LoadImages-77** 📁
Advanced image loader with batch processing and filtering capabilities.

**Features:**
- Single file or batch loading modes
- File filtering with wildcards (*.png, *logo*, etc.)
- Auto EXIF orientation correction
- Multiple resize modes (fit, fill, stretch)
- Subfolder scanning support
- Sorting options (name, date, size, random)

**Outputs:**
- `images`: Loaded image tensors
- `filenames`: List of loaded filenames
- `count`: Number of loaded images

### 2. **Canvas-77** 🖼️
Powerful canvas composer for layering multiple images with precise control.

**Features:**
- Layer-based composition system
- JSON-based layer positioning and properties
- Individual layer controls (position, scale, rotation, opacity)
- Up to 10 layers support
- Alpha compositing
- Simplified version with individual parameter controls

**Outputs:**
- `composed_image`: Final composited image

### 3. **Outline-77** 🎨
Creates customizable outlines from images with advanced morph