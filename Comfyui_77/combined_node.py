import torch
from PIL import Image, ImageEnhance
import numpy as np

class CombinedNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "gradient_color": ("COLOR",),
                "sharpness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "color_balance_red": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0}),
                "color_balance_green": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0}),
                "color_balance_blue": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0}),
                "hue": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine"
    CATEGORY = "ALL77"

    def combine(self, image, gradient_color, sharpness, color_balance_red, color_balance_green, color_balance_blue, hue, saturation):
        # Convert gradient color to image
        gradient_image = self.create_gradient_image(gradient_color, image[0].shape[1], image[0].shape[0])

        # Apply gradient color
        image = self.apply_gradient(image, gradient_image)

        # Apply sharpness
        image = self.apply_sharpness(image, sharpness)

        # Apply color balance
        image = self.apply_color_balance(image, color_balance_red, color_balance_green, color_balance_blue)

        # Apply hue/saturation
        image = self.apply_hue_saturation(image, hue, saturation)

        return (image,)

    def create_gradient_image(self, color, width, height):
        # Create a gradient image based on the given color
        # This is a placeholder; the actual implementation would depend on the desired gradient effect
        image = Image.new("RGB", (width, height), color)
        image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        return image

    def apply_gradient(self, image, gradient_image):
        # Apply the gradient image to the input image
        # Convert the image to PIL format
        pil_image = Image.fromarray(np.clip(image[0].cpu().numpy() * 255, 0, 255).astype(np.uint8).squeeze())
        pil_gradient = Image.fromarray(np.clip(gradient_image[0].cpu().numpy() * 255, 0, 255).astype(np.uint8).squeeze())

        # Blend the images
        blended_image = Image.blend(pil_image, pil_gradient, 0.5)

        # Convert back to tensor
        blended_image = torch.from_numpy(np.array(blended_image).astype(np.float32) / 255.0).unsqueeze(0)
        return blended_image

    def apply_sharpness(self, image, sharpness):
        # Apply sharpness to the image
        # Convert the image to PIL format
        pil_image = Image.fromarray(np.clip(image[0].cpu().numpy() * 255, 0, 255).astype(np.uint8).squeeze())

        # Apply sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        sharpened_image = enhancer.enhance(sharpness)

        # Convert back to tensor
        sharpened_image = torch.from_numpy(np.array(sharpened_image).astype(np.float32) / 255.0).unsqueeze(0)
        return sharpened_image

    def apply_color_balance(self, image, red, green, blue):
        # Apply color balance to the image
        # Convert the image to PIL format
        pil_image = Image.fromarray(np.clip(image[0].cpu().numpy() * 255, 0, 255).astype(np.uint8).squeeze())

        # Apply color balance (very basic implementation)
        enhancer = ImageEnhance.Color(pil_image)
        balanced_image = enhancer.enhance(1.0 + red - green - blue)

        # Convert back to tensor
        balanced_image = torch.from_numpy(np.array(balanced_image).astype(np.float32) / 255.0).unsqueeze(0)
        return balanced_image

    def apply_hue_saturation(self, image, hue, saturation):
        # Apply hue/saturation adjustments to the image
        # Convert the image to PIL format
        pil_image = Image.fromarray(np.clip(image[0].cpu().numpy() * 255, 0, 255).astype(np.uint8).squeeze())

        # Convert to HSV
        hsv_image = pil_image.convert("HSV")
        hsv_pixels = hsv_image.load()

        if hsv_pixels is None:
            return image

        # Adjust hue and saturation
        for x in range(hsv_image.width):
            for y in range(hsv_image.height):
                h, s, v = hsv_pixels[x, y]
                h = int((h + hue) % 256)  # Hue is 0-255 in PIL
                s = int(s * saturation)
                s = min(255, max(0, s))
                hsv_pixels[x, y] = (h, s, v)

        # Convert back to RGB
        rgb_image = hsv_image.convert("RGB")

        # Convert back to tensor
        rgb_image = torch.from_numpy(np.array(rgb_image).astype(np.float32) / 255.0).unsqueeze(0)
        return rgb_image

NODE_CLASS_MAPPINGS = {
    "CombinedNode": CombinedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CombinedNode": "Combined Node"
}
