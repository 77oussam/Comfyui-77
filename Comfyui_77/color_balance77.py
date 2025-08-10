import torch
import numpy as np
from PIL import Image

class ColorBalance77:
    """
    A ComfyUI node that replicates the Color Balance adjustment layer from Photoshop,
    allowing for separate color adjustments in shadows, midtones, and highlights.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tone_range": (["Shadows", "Midtones", "Highlights"],),
                "cyan_red": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "magenta_green": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "yellow_blue": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "preserve_luminosity": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "balance_color"
    CATEGORY = "ALL-77"

    def tensor_to_np(self, tensor):
        # Converts a torch tensor (B, H, W, C) to a list of NumPy arrays
        return [(img.cpu().numpy() * 255).astype(np.uint8) for img in tensor]

    def np_to_tensor(self, np_images):
        # Converts a list of NumPy arrays to a torch tensor (B, H, W, C)
        return torch.stack([torch.from_numpy(img.astype(np.float32) / 255.0) for img in np_images])

    def balance_color(self, image, tone_range, cyan_red, magenta_green, yellow_blue, preserve_luminosity):
        np_images = self.tensor_to_np(image)
        processed_images = []

        # Define the color adjustments. Note the inversion for green and blue.
        # Positive values move towards Red, Magenta, Yellow
        # Negative values move towards Cyan, Green, Blue
        color_adjust = np.array([cyan_red, magenta_green, yellow_blue]) * 100

        for img_np in np_images:
            # Create a floating point copy for calculations
            img_float = img_np.astype(np.float32)

            # Calculate luminance
            luminance = np.dot(img_float[..., :3], [0.299, 0.587, 0.114])

            # Create tone mask
            if tone_range == "Shadows":
                mask = luminance < 85
            elif tone_range == "Midtones":
                mask = (luminance >= 85) & (luminance < 170)
            else:  # Highlights
                mask = luminance >= 170
            
            if not np.any(mask):
                processed_images.append(img_np)
                continue

            # Store original luminance if needed
            if preserve_luminosity:
                original_luma = np.copy(luminance[mask])

            # Apply color adjustments to the selected region
            region = img_float[mask]
            
            # Add/Subtract color from channels
            # Red-Cyan
            region[:, 0] += color_adjust[0]
            # Green-Magenta
            region[:, 1] -= color_adjust[1]
            # Blue-Yellow
            region[:, 2] -= color_adjust[2]

            # Clip to valid range
            np.clip(region, 0, 255, out=region)

            # Preserve luminosity if enabled
            if preserve_luminosity:
                new_luma = np.dot(region, [0.299, 0.587, 0.114])
                luma_diff = original_luma - new_luma
                
                # Distribute the luminance difference back to the channels
                region += luma_diff[:, np.newaxis]
                np.clip(region, 0, 255, out=region)

            # Place the modified region back into the image
            img_float[mask] = region
            processed_images.append(img_float.astype(np.uint8))

        return (self.np_to_tensor(processed_images),)

NODE_CLASS_MAPPINGS = {
    "ColorBalance77": ColorBalance77
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorBalance77": "🎚️ Color Balance 77"
}
