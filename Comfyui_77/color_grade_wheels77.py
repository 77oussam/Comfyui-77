import torch
import numpy as np

class ColorGradeWheels77:
    """
    A ComfyUI node for three-way color grading using color wheels, inspired by professional tools.
    It provides separate color controls for shadows, midtones, and highlights.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_color": ("COLOR", {"label": "Shadow", "default": "#404040"}),
                "midtone_color": ("COLOR", {"label": "Midtone", "default": "#808080"}),
                "highlight_color": ("COLOR", {"label": "Highlight", "default": "#C0C0C0"}),
                "global_saturation": ("FLOAT", {"label": "Saturation", "default": 1.0, "min": 0.0, "max": 2.0}),
                "preserve_luminosity": ("BOOLEAN", {"label": "Preserve Luminosity", "default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_grade"
    CATEGORY = "ALL-77"

    def tensor_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().numpy()

    def np_to_tensor(self, np_array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np_array)

    def hex_to_rgb(self, hex_color: str) -> tuple[float, float, float]:
        hex_color = hex_color.lstrip('#')
        return (int(hex_color[0:2], 16) / 255.0, int(hex_color[2:4], 16) / 255.0, int(hex_color[4:6], 16) / 255.0)

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

    def apply_color_grade(self, image, shadow_color, midtone_color, highlight_color, global_saturation, preserve_luminosity):
        batch_np = self.tensor_to_np(image)
        processed_images = []

        for img_np in batch_np:
            original_luma = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]) if preserve_luminosity else None

            hsv = self.rgb_to_hsv(img_np)
            
            luminance = hsv[..., 2]
            shadows_mask = luminance < 0.3
            midtones_mask = (luminance >= 0.3) & (luminance <= 0.7)
            highlights_mask = luminance > 0.7
            
            for mask, hex_color in zip([shadows_mask, midtones_mask, highlights_mask],
                                       [shadow_color, midtone_color, highlight_color]):
                if not np.any(mask):
                    continue

                target_rgb = self.hex_to_rgb(hex_color)
                target_hsv = self.rgb_to_hsv(np.array(target_rgb))
                
                # Apply hue and saturation from the target color
                hsv[mask, 0] = target_hsv[0]
                hsv[mask, 1] = target_hsv[1]

            if global_saturation != 1.0:
                hsv[..., 1] *= global_saturation
            
            np.clip(hsv[..., 1], 0, 1, out=hsv[..., 1])

            graded_rgb = self.hsv_to_rgb(hsv)
            
            if preserve_luminosity and original_luma is not None:
                graded_luma = np.dot(graded_rgb[..., :3], [0.299, 0.587, 0.114])
                luma_diff = original_luma - graded_luma
                graded_rgb += luma_diff[..., np.newaxis]

            processed_images.append(np.clip(graded_rgb, 0, 1))

        return (self.np_to_tensor(np.array(processed_images)),)

NODE_CLASS_MAPPINGS = {
    "ColorGradeWheels77": ColorGradeWheels77
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorGradeWheels77": "🎛️ Color Grade Wheels 77"
}
