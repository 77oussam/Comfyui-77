import numpy as np
import torch
from PIL import Image

class MergeImages77:
    """
    Merge multiple input images into one single output image.
    User can select how many images to use (1-10).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_images": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Number of images to merge"
                }),
                "merge_mode": (["horizontal", "vertical", "grid"], {
                    "default": "horizontal",
                    "tooltip": "How to merge images"
                })
            },
            "optional": {
                **{f"image_{i+1}": ("IMAGE",) for i in range(10)}
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)
    FUNCTION = "merge_images"
    CATEGORY = "ALL-77"

    def comfy_tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        np_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        if np_array.shape[2] == 1:
            np_array = np.repeat(np_array, 3, axis=2)
        return Image.fromarray(np_array)

    def pil_to_comfy_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor."""
        np_array = np.array(pil_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_array).unsqueeze(0)
        return tensor

    def merge_images(self, num_images, merge_mode, **kwargs):
        images = []
        for i in range(num_images):
            img_key = f"image_{i+1}"
            if img_key in kwargs and kwargs[img_key] is not None:
                images.append(self.comfy_tensor_to_pil(kwargs[img_key]))

        if not images:
            raise ValueError("No images provided.")

        if merge_mode == "horizontal":
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            merged = Image.new("RGBA", (total_width, max_height), (0,0,0,0))
            x_offset = 0
            for img in images:
                merged.paste(img, (x_offset, 0), img if img.mode == "RGBA" else None)
                x_offset += img.width

        elif merge_mode == "vertical":
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)
            merged = Image.new("RGBA", (max_width, total_height), (0,0,0,0))
            y_offset = 0
            for img in images:
                merged.paste(img, (0, y_offset), img if img.mode == "RGBA" else None)
                y_offset += img.height

        elif merge_mode == "grid":
            grid_size = int(np.ceil(np.sqrt(len(images))))
            cell_width = max(img.width for img in images)
            cell_height = max(img.height for img in images)
            merged = Image.new("RGBA", (cell_width * grid_size, cell_height * grid_size), (0,0,0,0))
            for idx, img in enumerate(images):
                row = idx // grid_size
                col = idx % grid_size
                merged.paste(img, (col * cell_width, row * cell_height), img if img.mode == "RGBA" else None)

        else:
            raise ValueError("Unknown merge mode")

        return (self.pil_to_comfy_tensor(merged),)


NODE_CLASS_MAPPINGS = {
    "MergeImages-77": MergeImages77
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MergeImages-77": "Merge Images-77 🖼️"
}
