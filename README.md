# Comfyui_77 – Custom Node Collection

This folder contains **77-series** ComfyUI custom nodes, built for extended image editing, color grading, and compositing workflows.  
All nodes follow ComfyUI's standard node architecture and are designed to work in both CPU and GPU pipelines.

---

## 📦 Included Nodes

| File | Display Name | Description |
|------|--------------|-------------|
| `color_balance77.py` | 🎨 Color Balance 77 | Adjust shadows, midtones, and highlights like Photoshop's Color Balance panel. |
| `color_grade_wheels77.py` | 🎛️ Color Grade Wheels 77 | DaVinci Resolve–style Lift, Gamma, Gain color wheels. |
| `curve77.py` | 📈 Curve 77 | Editable tone curve for precise brightness/contrast and color adjustments. |
| `gradient_map77.py` | 🌈 Gradient Map 77 | Map image luminance to custom gradient colors. |
| `hue_saturation_lightness77.py` | 🎨 HSL 77 | Hue, saturation, and lightness adjustments similar to Photoshop. |
| `mergeimages77.py` | 🖼️ Merge Images 77 | Merge multiple images with adjustable blend modes and opacity. |
| `outline77.py` | ✏️ Outline 77 | Generate outlines from an image with color and thickness control. |
| `processing.py` | ⚙️ Processing Helpers | Internal utilities for image conversion, blending, and tensor handling. |
| `worker.py` | 🖥️ Worker | Optional async tasks for heavy processing jobs. |
| `combined_node.py` | 🧩 Combined Node | Merges multiple adjustments into a single node for faster workflows. |
| `main.py` | 🔧 Loader | Registers all nodes and their display names with ComfyUI. |
| `requirements.txt` | 📄 Requirements | Python dependencies for these nodes. |

---

## ⚡ Installation

1. Place the folder **`Comfyui_77`** into:
ComfyUI/custom_nodes/

go
Copy
Edit
2. Install dependencies:
```bash
pip install -r requirements.txt
Restart ComfyUI.

🖱️ Usage
Open ComfyUI.

Search for any of the listed nodes in the node search panel.

Connect an IMAGE input to the desired node.

Adjust parameters in real time.

Chain multiple nodes for advanced edits.

🔹 Tips
For Curve 77, click the curve editor to add, move, or remove points.

Color Grade Wheels 77 works best in sRGB space; consider converting before/after for other color spaces.

You can combine Merge Images 77 with Outline 77 to create stylized composites.

Processing.py contains reusable functions for anyone developing new custom nodes.

📜 License
This collection is provided as-is under the MIT License.
Feel free to modify and integrate into your own workflows.

📩 Contact
# Comfyui_77 – Custom Node Collection

This folder can contain **77-series** ComfyUI custom nodes, built for extended imag

yaml
Copy
Edit

---

If you want, I can also make a **second section** in this README specifically for your upcoming **`Layer_Studio_77`** node, so it’s documented and ready when you add it. That way, all your 77-series tools are documented in one place.






