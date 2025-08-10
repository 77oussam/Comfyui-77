"""
Outline-77 ComfyUI Node - Complete File for v3.30.4
Save this as: outline77.py
"""

import numpy as np
import cv2
import torch
from PIL import Image


class Outline77:
    """
    Creates customizable outlines from images with advanced morphological operations.
    Optimized for ComfyUI v3.30.4 tensor format and API requirements.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image. Will automatically detect and use alpha channel or create mask from content."
                }),
                "outline_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                    "tooltip": "Hex color for outline (e.g., #FFFFFF for white, #FF0000 for red)"
                }),
                "outline_width": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Thickness of outline in pixels"
                }),
                "softness": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Gaussian blur for outline softening (0=sharp, higher=softer)"
                }),
                "grow_shrink": ("INT", {
                    "default": 0,
                    "min": -64,
                    "max": 64,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Grow (+) or shrink (-) mask before outline extraction"
                }),
                "round_corners": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply morphological opening to smooth corners"
                }),
                "keep_alpha": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve original image inside outline (sticker effect)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("outlined_image",)
    FUNCTION = "create_outline"
    CATEGORY = "ALL-77"
    
    # ComfyUI v3.30.4 compatibility
    OUTPUT_NODE = False
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB with robust validation."""
        try:
            hex_color = hex_color.strip().lstrip('#')
            if len(hex_color) == 3:
                # Convert 3-digit hex to 6-digit
                hex_color = ''.join([c*2 for c in hex_color])
            if len(hex_color) != 6:
                return (255, 255, 255)  # Default white
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except:
            return (255, 255, 255)  # Fallback to white
    
    def comfy_tensor_to_pil(self, tensor):
        """Convert ComfyUI v3.30.4 tensor to PIL Image."""
        # ComfyUI v3.30.4 format: [batch, height, width, channels] in range [0,1]
        if tensor.dim() == 4:
            # Remove batch dimension
            tensor = tensor[0]
        
        # Convert from [0,1] to [0,255]
        np_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Handle different channel counts
        if np_array.shape[2] == 1:
            # Grayscale to RGB
            np_array = np.repeat(np_array, 3, axis=2)
            return Image.fromarray(np_array, 'RGB')
        elif np_array.shape[2] == 3:
            # RGB
            return Image.fromarray(np_array, 'RGB')
        elif np_array.shape[2] == 4:
            # RGBA
            return Image.fromarray(np_array, 'RGBA')
        else:
            # Fallback - take first 3 channels
            return Image.fromarray(np_array[:, :, :3], 'RGB')
    
    def pil_to_comfy_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI v3.30.4 tensor format."""
        # Ensure RGBA for consistent output
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        # Convert to numpy array
        np_array = np.array(pil_image, dtype=np.float32)
        
        # Normalize to [0,1] range
        np_array = np_array / 255.0
        
        # Convert to tensor: [height, width, channels] -> [batch, height, width, channels]
        tensor = torch.from_numpy(np_array).unsqueeze(0)
        
        return tensor
    
    def create_alpha_mask(self, pil_image):
        """Create alpha mask from image content."""
        if pil_image.mode == 'RGBA':
            # Use existing alpha channel
            alpha = np.array(pil_image)[:, :, 3]
        else:
            # Create alpha from non-background pixels
            img_array = np.array(pil_image.convert('RGB'))
            
            # Detect background (assume corners are background)
            h, w = img_array.shape[:2]
            corner_colors = [
                img_array[0, 0],           # top-left
                img_array[0, w-1],         # top-right  
                img_array[h-1, 0],         # bottom-left
                img_array[h-1, w-1]        # bottom-right
            ]
            
            # Use most common corner color as background
            bg_color = corner_colors[0]  # Simple approach
            
            # Create mask where pixels differ significantly from background
            diff = np.sum(np.abs(img_array - bg_color), axis=2)
            alpha = np.where(diff > 30, 255, 0).astype(np.uint8)
        
        return alpha
    
    def create_outline(self, image, outline_color, outline_width, softness, grow_shrink, round_corners, keep_alpha):
        """
        Create outlined image optimized for ComfyUI v3.30.4.
        """
        try:
            # Convert ComfyUI tensor to PIL
            pil_image = self.comfy_tensor_to_pil(image)
            
            # Get dimensions
            width, height = pil_image.size
            
            # Create alpha mask
            alpha = self.create_alpha_mask(pil_image)
            
            # Step 1: Binary mask from alpha
            binary_mask = (alpha > 127).astype(np.uint8) * 255
            
            # Step 2: Grow/shrink mask if needed
            if grow_shrink != 0:
                kernel_size = abs(grow_shrink)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size*2+1, kernel_size*2+1))
                
                if grow_shrink > 0:
                    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
                else:
                    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
            
            # Step 3: Create outline edge
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            inner_mask = cv2.erode(binary_mask, erode_kernel, iterations=1)
            outline_edge = cv2.bitwise_xor(binary_mask, inner_mask)
            
            # Step 4: Expand outline to desired width
            if outline_width > 1:
                width_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outline_width*2+1, outline_width*2+1))
                outline_edge = cv2.dilate(outline_edge, width_kernel, iterations=1)
            
            # Step 5: Round corners if requested
            if round_corners:
                round_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                outline_edge = cv2.morphologyEx(outline_edge, cv2.MORPH_OPEN, round_kernel)
            
            # Step 6: Apply softness
            if softness > 0:
                # Ensure odd kernel size
                blur_kernel = softness * 2 + 1
                if blur_kernel > 99:  # Cap for performance
                    blur_kernel = 99
                outline_edge = cv2.GaussianBlur(outline_edge, (blur_kernel, blur_kernel), softness/3.0)
            
            # Step 7: Create colored outline
            outline_rgb = self.hex_to_rgb(outline_color)
            
            # Initialize final RGBA image
            final_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Apply outline color
            edge_strength = outline_edge.astype(np.float32) / 255.0
            for channel in range(3):
                final_rgba[:, :, channel] = (edge_strength * outline_rgb[channel]).astype(np.uint8)
            
            # Set outline alpha
            final_rgba[:, :, 3] = outline_edge
            
            # Step 8: Composite with original if keep_alpha enabled
            if keep_alpha:
                original_rgba = pil_image.convert('RGBA')
                orig_array = np.array(original_rgba)
                
                # Mask original to shape
                orig_masked = orig_array.copy()
                orig_masked[:, :, 3] = np.minimum(orig_masked[:, :, 3], binary_mask)
                
                # Alpha blend outline and original
                outline_alpha = final_rgba[:, :, 3].astype(np.float32) / 255.0
                original_alpha = orig_masked[:, :, 3].astype(np.float32) / 255.0
                
                # Combined alpha
                combined_alpha = outline_alpha + original_alpha * (1 - outline_alpha)
                
                # Blend RGB channels
                for channel in range(3):
                    final_rgba[:, :, channel] = np.where(
                        combined_alpha > 0,
                        (final_rgba[:, :, channel].astype(np.float32) * outline_alpha + 
                         orig_masked[:, :, channel].astype(np.float32) * original_alpha * (1 - outline_alpha)) / combined_alpha,
                        0
                    ).astype(np.uint8)
                
                final_rgba[:, :, 3] = (combined_alpha * 255).astype(np.uint8)
            
            # Convert back to PIL and then ComfyUI tensor
            result_pil = Image.fromarray(final_rgba, 'RGBA')
            result_tensor = self.pil_to_comfy_tensor(result_pil)
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"Outline77 Error: {str(e)}")
            # Return transparent fallback image
            fallback = torch.zeros((1, 512, 512, 4), dtype=torch.float32)
            return (fallback,)


# ComfyUI v3.30.4 Registration
NODE_CLASS_MAPPINGS = {
    "Outline77": Outline77
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Outline77": "Outline-77"
}
