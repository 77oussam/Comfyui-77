import torch
import numpy as np

class GradientMap77:
    """
    Applies a color gradient to an image based on its luminance,
    similar to Photoshop's "Gradient Map" adjustment layer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "start_color": ("COLOR", {"default": "#000000"}),
                "end_color": ("COLOR", {"default": "#FFFFFF"}),
                "blend_mode": (["Normal", "Overlay", "Soft Light", "Color"],),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_gradient_map"
    CATEGORY = "ALL-77"

    def tensor_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().numpy()

    def np_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array)

    def hex_to_rgb(self, hex_color: str) -> np.ndarray:
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0

    # Blending functions
    def blend_normal(self, base, blend):
        return blend

    def blend_overlay(self, base, blend):
        return np.where(base <= 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend))

    def blend_soft_light(self, base, blend):
        return np.where(blend <= 0.5, base - (1 - 2 * blend) * base * (1 - base), base + (2 * blend - 1) * (((np.sqrt(base) if np.all(base >= 0) else base) if np.all(base >= 0) else base) - base))

    def blend_color(self, base, blend):
        base_hsv = self.rgb_to_hsv(base)
        blend_hsv = self.rgb_to_hsv(blend)
        result_hsv = np.stack([blend_hsv[..., 0], blend_hsv[..., 1], base_hsv[..., 2]], axis=-1)
        return self.hsv_to_rgb(result_hsv)
        
    def rgb_to_hsv(self, rgb):
        v = np.max(rgb, axis=-1)
        old_min = np.min(rgb, axis=-1)
        delta = v - old_min
        s = np.where(v != 0, delta / v, 0)
        h = np.zeros_like(v)
        non_zero_delta = delta != 0
        idx_r = (v == rgb[..., 0]) & non_zero_delta
        h[idx_r] = (rgb[idx_r, 1] - rgb[idx_r, 2]) / delta[idx_r]
        idx_g = (v == rgb[..., 1]) & non_zero_delta
        h[idx_g] = 2 + (rgb[idx_g, 2] - rgb[idx_g, 0]) / delta[idx_g]
        idx_b = (v == rgb[..., 2]) & non_zero_delta
        h[idx_b] = 4 + (rgb[idx_b, 0] - rgb[idx_b, 1]) / delta[idx_b]
        h = (h / 6) % 1.0
        return np.stack([h, s, v], axis=-1)

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        rgb = np.zeros_like(hsv)
        idx = i % 6 == 0
        rgb[idx] = np.stack([v[idx], t[idx], p[idx]], axis=-1)
        idx = i == 1
        rgb[idx] = np.stack([q[idx], v[idx], p[idx]], axis=-1)
        idx = i == 2
        rgb[idx] = np.stack([p[idx], v[idx], t[idx]], axis=-1)
        idx = i == 3
        rgb[idx] = np.stack([p[idx], q[idx], v[idx]], axis=-1)
        idx = i == 4
        rgb[idx] = np.stack([t[idx], p[idx], v[idx]], axis=-1)
        idx = i == 5
        rgb[idx] = np.stack([v[idx], p[idx], q[idx]], axis=-1)
        return rgb

    def apply_gradient_map(self, image, start_color, end_color, blend_mode, opacity):
        batch_np = self.tensor_to_np(image)
        processed_images = []

        start_rgb = self.hex_to_rgb(start_color)
        end_rgb = self.hex_to_rgb(end_color)

        for img_np in batch_np:
            luminance = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
            
            gradient_map = np.linspace(start_rgb, end_rgb, 256).transpose()
            
            # Create an index from the luminance
            lum_index = (luminance * 255).astype(np.uint8)
            
            # Apply the gradient map
            gradient_img = np.array([
                gradient_map[0][lum_index],
                gradient_map[1][lum_index],
                gradient_map[2][lum_index]
            ]).transpose(1, 2, 0)

            # Select blend function
            blend_func = {
                "Normal": self.blend_normal,
                "Overlay": self.blend_overlay,
                "Soft Light": self.blend_soft_light,
                "Color": self.blend_color,
            }[blend_mode]

            blended_img = blend_func(img_np, gradient_img)
            
            # Apply opacity
            final_img = img_np * (1 - opacity) + blended_img * opacity
            processed_images.append(np.clip(final_img, 0, 1))

        return (self.np_to_tensor(np.array(processed_images)),)

NODE_CLASS_MAPPINGS = {
    "GradientMap77": GradientMap77
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GradientMap77": "🎨 Gradient Map 77"
}
