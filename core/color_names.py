"""
Named-color lookup.

Finds the closest human-readable name for any RGB color by comparing it
against the 140 standard CSS/HTML color names in the CIE L*a*b* color space.
L*a*b* is used instead of plain RGB because distances in L*a*b* match how
different two colors *look* to the human eye.
"""
import re

import cv2
import numpy as np

# Standard CSS / HTML named colors (duplicates such as Aqua/Fuchsia removed)
CSS_COLORS = {
    "AliceBlue": "#F0F8FF", "AntiqueWhite": "#FAEBD7", "Aquamarine": "#7FFFD4",
    "Azure": "#F0FFFF", "Beige": "#F5F5DC", "Bisque": "#FFE4C4", "Black": "#000000",
    "BlanchedAlmond": "#FFEBCD", "Blue": "#0000FF", "BlueViolet": "#8A2BE2",
    "Brown": "#A52A2A", "BurlyWood": "#DEB887", "CadetBlue": "#5F9EA0",
    "Chartreuse": "#7FFF00", "Chocolate": "#D2691E", "Coral": "#FF7F50",
    "CornflowerBlue": "#6495ED", "Cornsilk": "#FFF8DC", "Crimson": "#DC143C",
    "Cyan": "#00FFFF", "DarkBlue": "#00008B", "DarkCyan": "#008B8B",
    "DarkGoldenrod": "#B8860B", "DarkGray": "#A9A9A9", "DarkGreen": "#006400",
    "DarkKhaki": "#BDB76B", "DarkMagenta": "#8B008B", "DarkOliveGreen": "#556B2F",
    "DarkOrange": "#FF8C00", "DarkOrchid": "#9932CC", "DarkRed": "#8B0000",
    "DarkSalmon": "#E9967A", "DarkSeaGreen": "#8FBC8F", "DarkSlateBlue": "#483D8B",
    "DarkSlateGray": "#2F4F4F", "DarkTurquoise": "#00CED1", "DarkViolet": "#9400D3",
    "DeepPink": "#FF1493", "DeepSkyBlue": "#00BFFF", "DimGray": "#696969",
    "DodgerBlue": "#1E90FF", "FireBrick": "#B22222", "FloralWhite": "#FFFAF0",
    "ForestGreen": "#228B22", "Gainsboro": "#DCDCDC", "GhostWhite": "#F8F8FF",
    "Gold": "#FFD700", "Goldenrod": "#DAA520", "Gray": "#808080", "Green": "#008000",
    "GreenYellow": "#ADFF2F", "Honeydew": "#F0FFF0", "HotPink": "#FF69B4",
    "IndianRed": "#CD5C5C", "Indigo": "#4B0082", "Ivory": "#FFFFF0", "Khaki": "#F0E68C",
    "Lavender": "#E6E6FA", "LavenderBlush": "#FFF0F5", "LawnGreen": "#7CFC00",
    "LemonChiffon": "#FFFACD", "LightBlue": "#ADD8E6", "LightCoral": "#F08080",
    "LightCyan": "#E0FFFF", "LightGoldenrodYellow": "#FAFAD2", "LightGray": "#D3D3D3",
    "LightGreen": "#90EE90", "LightPink": "#FFB6C1", "LightSalmon": "#FFA07A",
    "LightSeaGreen": "#20B2AA", "LightSkyBlue": "#87CEFA", "LightSlateGray": "#778899",
    "LightSteelBlue": "#B0C4DE", "LightYellow": "#FFFFE0", "Lime": "#00FF00",
    "LimeGreen": "#32CD32", "Linen": "#FAF0E6", "Magenta": "#FF00FF", "Maroon": "#800000",
    "MediumAquamarine": "#66CDAA", "MediumBlue": "#0000CD", "MediumOrchid": "#BA55D3",
    "MediumPurple": "#9370DB", "MediumSeaGreen": "#3CB371", "MediumSlateBlue": "#7B68EE",
    "MediumSpringGreen": "#00FA9A", "MediumTurquoise": "#48D1CC",
    "MediumVioletRed": "#C71585", "MidnightBlue": "#191970", "MintCream": "#F5FFFA",
    "MistyRose": "#FFE4E1", "Moccasin": "#FFE4B5", "NavajoWhite": "#FFDEAD",
    "Navy": "#000080", "OldLace": "#FDF5E6", "Olive": "#808000", "OliveDrab": "#6B8E23",
    "Orange": "#FFA500", "OrangeRed": "#FF4500", "Orchid": "#DA70D6",
    "PaleGoldenrod": "#EEE8AA", "PaleGreen": "#98FB98", "PaleTurquoise": "#AFEEEE",
    "PaleVioletRed": "#DB7093", "PapayaWhip": "#FFEFD5", "PeachPuff": "#FFDAB9",
    "Peru": "#CD853F", "Pink": "#FFC0CB", "Plum": "#DDA0DD", "PowderBlue": "#B0E0E6",
    "Purple": "#800080", "RebeccaPurple": "#663399", "Red": "#FF0000",
    "RosyBrown": "#BC8F8F", "RoyalBlue": "#4169E1", "SaddleBrown": "#8B4513",
    "Salmon": "#FA8072", "SandyBrown": "#F4A460", "SeaGreen": "#2E8B57",
    "Seashell": "#FFF5EE", "Sienna": "#A0522D", "Silver": "#C0C0C0", "SkyBlue": "#87CEEB",
    "SlateBlue": "#6A5ACD", "SlateGray": "#708090", "Snow": "#FFFAFA",
    "SpringGreen": "#00FF7F", "SteelBlue": "#4682B4", "Tan": "#D2B48C", "Teal": "#008080",
    "Thistle": "#D8BFD8", "Tomato": "#FF6347", "Turquoise": "#40E0D0", "Violet": "#EE82EE",
    "Wheat": "#F5DEB3", "White": "#FFFFFF", "WhiteSmoke": "#F5F5F5", "Yellow": "#FFFF00",
    "YellowGreen": "#9ACD32",
}


def _pretty(name: str) -> str:
    """'DarkSlateGray' -> 'Dark Slate Gray'"""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name)


def _hex_to_rgb(hex_code: str):
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 RGB -> (N, 3) float32 L*a*b*."""
    img = (rgb_array.reshape(-1, 1, 3).astype(np.float32)) / 255.0
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3)


_NAMES = [_pretty(n) for n in CSS_COLORS]
_RGB = np.array([_hex_to_rgb(h) for h in CSS_COLORS.values()], dtype=np.uint8)
_LAB = _rgb_to_lab(_RGB)


def nearest_color_name(rgb) -> str:
    """Return the closest named color for an (R, G, B) tuple."""
    lab = _rgb_to_lab(np.array([rgb], dtype=np.uint8))[0]
    distances = np.sum((_LAB - lab) ** 2, axis=1)
    return _NAMES[int(np.argmin(distances))]
