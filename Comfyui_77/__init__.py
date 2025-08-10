from .color_balance77 import NODE_CLASS_MAPPINGS as color_balance_mappings, NODE_DISPLAY_NAME_MAPPINGS as color_balance_display_mappings
from .color_grade_wheels77 import NODE_CLASS_MAPPINGS as color_grade_wheels_mappings, NODE_DISPLAY_NAME_MAPPINGS as color_grade_wheels_display_mappings
from .combined_node import NODE_CLASS_MAPPINGS as combined_node_mappings, NODE_DISPLAY_NAME_MAPPINGS as combined_node_display_mappings
from .curve77 import NODE_CLASS_MAPPINGS as curve77_mappings, NODE_DISPLAY_NAME_MAPPINGS as curve77_display_mappings
from .gradient_map77 import NODE_CLASS_MAPPINGS as gradient_map_mappings, NODE_DISPLAY_NAME_MAPPINGS as gradient_map_display_mappings
from .hue_saturation_lightness77 import NODE_CLASS_MAPPINGS as hue_saturation_lightness_mappings, NODE_DISPLAY_NAME_MAPPINGS as hue_saturation_lightness_display_mappings
from .mergeimages77 import NODE_CLASS_MAPPINGS as mergeimages77_mappings, NODE_DISPLAY_NAME_MAPPINGS as mergeimages77_display_mappings
from .outline77 import NODE_CLASS_MAPPINGS as outline77_mappings, NODE_DISPLAY_NAME_MAPPINGS as outline77_display_mappings

NODE_CLASS_MAPPINGS = {
    **color_balance_mappings,
    **color_grade_wheels_mappings,
    **combined_node_mappings,
    **curve77_mappings,
    **gradient_map_mappings,
    **hue_saturation_lightness_mappings,
    **mergeimages77_mappings,
    **outline77_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **color_balance_display_mappings,
    **color_grade_wheels_display_mappings,
    **combined_node_display_mappings,
    **curve77_display_mappings,
    **gradient_map_display_mappings,
    **hue_saturation_lightness_display_mappings,
    **mergeimages77_display_mappings,
    **outline77_display_mappings,
}

WEB_DIRECTORY = "web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
