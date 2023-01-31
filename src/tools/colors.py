#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2012 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2012-05-12.

"""
Colors utilities
================


"""

import re
import numpy as np


def main():
    """Generates code for name_to_rgb dict, assuming an rgb.txt file available (in X11 format)."""
    with open('rgb.txt') as fp:

        line = fp.readline()
        while line:
            reg = re.match('\s*(\d+)\s*(\d+)\s*(\d+)\s*(\w.*\w).*', line)
            if reg:
                r = int(reg.group(1)) / 255.
                g = int(reg.group(2)) / 255.
                b = int(reg.group(3)) / 255.
                d = reg.group(4)
                print("'%s': np.array([%.2f, %.2f, %.2f])," % (d, r, g, b))
            line = fp.readline()


name_to_rgb = {
    'snow': np.array([1.00, 0.98, 0.98]),
    'ghost white': np.array([0.97, 0.97, 1.00]),
    'GhostWhite': np.array([0.97, 0.97, 1.00]),
    'white smoke': np.array([0.96, 0.96, 0.96]),
    'WhiteSmoke': np.array([0.96, 0.96, 0.96]),
    'gainsboro': np.array([0.86, 0.86, 0.86]),
    'floral white': np.array([1.00, 0.98, 0.94]),
    'FloralWhite': np.array([1.00, 0.98, 0.94]),
    'old lace': np.array([0.99, 0.96, 0.90]),
    'OldLace': np.array([0.99, 0.96, 0.90]),
    'linen': np.array([0.98, 0.94, 0.90]),
    'antique white': np.array([0.98, 0.92, 0.84]),
    'AntiqueWhite': np.array([0.98, 0.92, 0.84]),
    'papaya whip': np.array([1.00, 0.94, 0.84]),
    'PapayaWhip': np.array([1.00, 0.94, 0.84]),
    'blanched almond': np.array([1.00, 0.92, 0.80]),
    'BlanchedAlmond': np.array([1.00, 0.92, 0.80]),
    'bisque': np.array([1.00, 0.89, 0.77]),
    'peach puff': np.array([1.00, 0.85, 0.73]),
    'PeachPuff': np.array([1.00, 0.85, 0.73]),
    'navajo white': np.array([1.00, 0.87, 0.68]),
    'NavajoWhite': np.array([1.00, 0.87, 0.68]),
    'moccasin': np.array([1.00, 0.89, 0.71]),
    'cornsilk': np.array([1.00, 0.97, 0.86]),
    'ivory': np.array([1.00, 1.00, 0.94]),
    'lemon chiffon': np.array([1.00, 0.98, 0.80]),
    'LemonChiffon': np.array([1.00, 0.98, 0.80]),
    'seashell': np.array([1.00, 0.96, 0.93]),
    'honeydew': np.array([0.94, 1.00, 0.94]),
    'mint cream': np.array([0.96, 1.00, 0.98]),
    'MintCream': np.array([0.96, 1.00, 0.98]),
    'azure': np.array([0.94, 1.00, 1.00]),
    'alice blue': np.array([0.94, 0.97, 1.00]),
    'AliceBlue': np.array([0.94, 0.97, 1.00]),
    'lavender': np.array([0.90, 0.90, 0.98]),
    'lavender blush': np.array([1.00, 0.94, 0.96]),
    'LavenderBlush': np.array([1.00, 0.94, 0.96]),
    'misty rose': np.array([1.00, 0.89, 0.88]),
    'MistyRose': np.array([1.00, 0.89, 0.88]),
    'white': np.array([1.00, 1.00, 1.00]),
    'black': np.array([0.00, 0.00, 0.00]),
    'dark slate gray': np.array([0.18, 0.31, 0.31]),
    'DarkSlateGray': np.array([0.18, 0.31, 0.31]),
    'dark slate grey': np.array([0.18, 0.31, 0.31]),
    'DarkSlateGrey': np.array([0.18, 0.31, 0.31]),
    'dim gray': np.array([0.41, 0.41, 0.41]),
    'DimGray': np.array([0.41, 0.41, 0.41]),
    'dim grey': np.array([0.41, 0.41, 0.41]),
    'DimGrey': np.array([0.41, 0.41, 0.41]),
    'slate gray': np.array([0.44, 0.50, 0.56]),
    'SlateGray': np.array([0.44, 0.50, 0.56]),
    'slate grey': np.array([0.44, 0.50, 0.56]),
    'SlateGrey': np.array([0.44, 0.50, 0.56]),
    'light slate gray': np.array([0.47, 0.53, 0.60]),
    'LightSlateGray': np.array([0.47, 0.53, 0.60]),
    'light slate grey': np.array([0.47, 0.53, 0.60]),
    'LightSlateGrey': np.array([0.47, 0.53, 0.60]),
    'gray': np.array([0.75, 0.75, 0.75]),
    'grey': np.array([0.75, 0.75, 0.75]),
    'light grey': np.array([0.83, 0.83, 0.83]),
    'LightGrey': np.array([0.83, 0.83, 0.83]),
    'light gray': np.array([0.83, 0.83, 0.83]),
    'LightGray': np.array([0.83, 0.83, 0.83]),
    'midnight blue': np.array([0.10, 0.10, 0.44]),
    'MidnightBlue': np.array([0.10, 0.10, 0.44]),
    'navy': np.array([0.00, 0.00, 0.50]),
    'navy blue': np.array([0.00, 0.00, 0.50]),
    'NavyBlue': np.array([0.00, 0.00, 0.50]),
    'cornflower blue': np.array([0.39, 0.58, 0.93]),
    'CornflowerBlue': np.array([0.39, 0.58, 0.93]),
    'dark slate blue': np.array([0.28, 0.24, 0.55]),
    'DarkSlateBlue': np.array([0.28, 0.24, 0.55]),
    'slate blue': np.array([0.42, 0.35, 0.80]),
    'SlateBlue': np.array([0.42, 0.35, 0.80]),
    'medium slate blue': np.array([0.48, 0.41, 0.93]),
    'MediumSlateBlue': np.array([0.48, 0.41, 0.93]),
    'light slate blue': np.array([0.52, 0.44, 1.00]),
    'LightSlateBlue': np.array([0.52, 0.44, 1.00]),
    'medium blue': np.array([0.00, 0.00, 0.80]),
    'MediumBlue': np.array([0.00, 0.00, 0.80]),
    'royal blue': np.array([0.25, 0.41, 0.88]),
    'RoyalBlue': np.array([0.25, 0.41, 0.88]),
    'blue': np.array([0.00, 0.00, 1.00]),
    'dodger blue': np.array([0.12, 0.56, 1.00]),
    'DodgerBlue': np.array([0.12, 0.56, 1.00]),
    'deep sky blue': np.array([0.00, 0.75, 1.00]),
    'DeepSkyBlue': np.array([0.00, 0.75, 1.00]),
    'sky blue': np.array([0.53, 0.81, 0.92]),
    'SkyBlue': np.array([0.53, 0.81, 0.92]),
    'light sky blue': np.array([0.53, 0.81, 0.98]),
    'LightSkyBlue': np.array([0.53, 0.81, 0.98]),
    'steel blue': np.array([0.27, 0.51, 0.71]),
    'SteelBlue': np.array([0.27, 0.51, 0.71]),
    'light steel blue': np.array([0.69, 0.77, 0.87]),
    'LightSteelBlue': np.array([0.69, 0.77, 0.87]),
    'light blue': np.array([0.68, 0.85, 0.90]),
    'LightBlue': np.array([0.68, 0.85, 0.90]),
    'powder blue': np.array([0.69, 0.88, 0.90]),
    'PowderBlue': np.array([0.69, 0.88, 0.90]),
    'pale turquoise': np.array([0.69, 0.93, 0.93]),
    'PaleTurquoise': np.array([0.69, 0.93, 0.93]),
    'dark turquoise': np.array([0.00, 0.81, 0.82]),
    'DarkTurquoise': np.array([0.00, 0.81, 0.82]),
    'medium turquoise': np.array([0.28, 0.82, 0.80]),
    'MediumTurquoise': np.array([0.28, 0.82, 0.80]),
    'turquoise': np.array([0.25, 0.88, 0.82]),
    'cyan': np.array([0.00, 1.00, 1.00]),
    'light cyan': np.array([0.88, 1.00, 1.00]),
    'LightCyan': np.array([0.88, 1.00, 1.00]),
    'cadet blue': np.array([0.37, 0.62, 0.63]),
    'CadetBlue': np.array([0.37, 0.62, 0.63]),
    'medium aquamarine': np.array([0.40, 0.80, 0.67]),
    'MediumAquamarine': np.array([0.40, 0.80, 0.67]),
    'aquamarine': np.array([0.50, 1.00, 0.83]),
    'dark green': np.array([0.00, 0.39, 0.00]),
    'DarkGreen': np.array([0.00, 0.39, 0.00]),
    'dark olive green': np.array([0.33, 0.42, 0.18]),
    'DarkOliveGreen': np.array([0.33, 0.42, 0.18]),
    'dark sea green': np.array([0.56, 0.74, 0.56]),
    'DarkSeaGreen': np.array([0.56, 0.74, 0.56]),
    'sea green': np.array([0.18, 0.55, 0.34]),
    'SeaGreen': np.array([0.18, 0.55, 0.34]),
    'medium sea green': np.array([0.24, 0.70, 0.44]),
    'MediumSeaGreen': np.array([0.24, 0.70, 0.44]),
    'light sea green': np.array([0.13, 0.70, 0.67]),
    'LightSeaGreen': np.array([0.13, 0.70, 0.67]),
    'pale green': np.array([0.60, 0.98, 0.60]),
    'PaleGreen': np.array([0.60, 0.98, 0.60]),
    'spring green': np.array([0.00, 1.00, 0.50]),
    'SpringGreen': np.array([0.00, 1.00, 0.50]),
    'lawn green': np.array([0.49, 0.99, 0.00]),
    'LawnGreen': np.array([0.49, 0.99, 0.00]),
    'green': np.array([0.00, 1.00, 0.00]),
    'chartreuse': np.array([0.50, 1.00, 0.00]),
    'medium spring green': np.array([0.00, 0.98, 0.60]),
    'MediumSpringGreen': np.array([0.00, 0.98, 0.60]),
    'green yellow': np.array([0.68, 1.00, 0.18]),
    'GreenYellow': np.array([0.68, 1.00, 0.18]),
    'lime green': np.array([0.20, 0.80, 0.20]),
    'LimeGreen': np.array([0.20, 0.80, 0.20]),
    'yellow green': np.array([0.60, 0.80, 0.20]),
    'YellowGreen': np.array([0.60, 0.80, 0.20]),
    'forest green': np.array([0.13, 0.55, 0.13]),
    'ForestGreen': np.array([0.13, 0.55, 0.13]),
    'olive drab': np.array([0.42, 0.56, 0.14]),
    'OliveDrab': np.array([0.42, 0.56, 0.14]),
    'dark khaki': np.array([0.74, 0.72, 0.42]),
    'DarkKhaki': np.array([0.74, 0.72, 0.42]),
    'khaki': np.array([0.94, 0.90, 0.55]),
    'pale goldenrod': np.array([0.93, 0.91, 0.67]),
    'PaleGoldenrod': np.array([0.93, 0.91, 0.67]),
    'light goldenrod yellow': np.array([0.98, 0.98, 0.82]),
    'LightGoldenrodYellow': np.array([0.98, 0.98, 0.82]),
    'light yellow': np.array([1.00, 1.00, 0.88]),
    'LightYellow': np.array([1.00, 1.00, 0.88]),
    'yellow': np.array([1.00, 1.00, 0.00]),
    'gold': np.array([1.00, 0.84, 0.00]),
    'light goldenrod': np.array([0.93, 0.87, 0.51]),
    'LightGoldenrod': np.array([0.93, 0.87, 0.51]),
    'goldenrod': np.array([0.85, 0.65, 0.13]),
    'dark goldenrod': np.array([0.72, 0.53, 0.04]),
    'DarkGoldenrod': np.array([0.72, 0.53, 0.04]),
    'rosy brown': np.array([0.74, 0.56, 0.56]),
    'RosyBrown': np.array([0.74, 0.56, 0.56]),
    'indian red': np.array([0.80, 0.36, 0.36]),
    'IndianRed': np.array([0.80, 0.36, 0.36]),
    'saddle brown': np.array([0.55, 0.27, 0.07]),
    'SaddleBrown': np.array([0.55, 0.27, 0.07]),
    'sienna': np.array([0.63, 0.32, 0.18]),
    'peru': np.array([0.80, 0.52, 0.25]),
    'burlywood': np.array([0.87, 0.72, 0.53]),
    'beige': np.array([0.96, 0.96, 0.86]),
    'wheat': np.array([0.96, 0.87, 0.70]),
    'sandy brown': np.array([0.96, 0.64, 0.38]),
    'SandyBrown': np.array([0.96, 0.64, 0.38]),
    'tan': np.array([0.82, 0.71, 0.55]),
    'chocolate': np.array([0.82, 0.41, 0.12]),
    'firebrick': np.array([0.70, 0.13, 0.13]),
    'brown': np.array([0.65, 0.16, 0.16]),
    'dark salmon': np.array([0.91, 0.59, 0.48]),
    'DarkSalmon': np.array([0.91, 0.59, 0.48]),
    'salmon': np.array([0.98, 0.50, 0.45]),
    'light salmon': np.array([1.00, 0.63, 0.48]),
    'LightSalmon': np.array([1.00, 0.63, 0.48]),
    'orange': np.array([1.00, 0.65, 0.00]),
    'dark orange': np.array([1.00, 0.55, 0.00]),
    'DarkOrange': np.array([1.00, 0.55, 0.00]),
    'coral': np.array([1.00, 0.50, 0.31]),
    'light coral': np.array([0.94, 0.50, 0.50]),
    'LightCoral': np.array([0.94, 0.50, 0.50]),
    'tomato': np.array([1.00, 0.39, 0.28]),
    'orange red': np.array([1.00, 0.27, 0.00]),
    'OrangeRed': np.array([1.00, 0.27, 0.00]),
    'red': np.array([1.00, 0.00, 0.00]),
    'hot pink': np.array([1.00, 0.41, 0.71]),
    'HotPink': np.array([1.00, 0.41, 0.71]),
    'deep pink': np.array([1.00, 0.08, 0.58]),
    'DeepPink': np.array([1.00, 0.08, 0.58]),
    'pink': np.array([1.00, 0.75, 0.80]),
    'light pink': np.array([1.00, 0.71, 0.76]),
    'LightPink': np.array([1.00, 0.71, 0.76]),
    'pale violet red': np.array([0.86, 0.44, 0.58]),
    'PaleVioletRed': np.array([0.86, 0.44, 0.58]),
    'maroon': np.array([0.69, 0.19, 0.38]),
    'medium violet red': np.array([0.78, 0.08, 0.52]),
    'MediumVioletRed': np.array([0.78, 0.08, 0.52]),
    'violet red': np.array([0.82, 0.13, 0.56]),
    'VioletRed': np.array([0.82, 0.13, 0.56]),
    'magenta': np.array([1.00, 0.00, 1.00]),
    'violet': np.array([0.93, 0.51, 0.93]),
    'plum': np.array([0.87, 0.63, 0.87]),
    'orchid': np.array([0.85, 0.44, 0.84]),
    'medium orchid': np.array([0.73, 0.33, 0.83]),
    'MediumOrchid': np.array([0.73, 0.33, 0.83]),
    'dark orchid': np.array([0.60, 0.20, 0.80]),
    'DarkOrchid': np.array([0.60, 0.20, 0.80]),
    'dark violet': np.array([0.58, 0.00, 0.83]),
    'DarkViolet': np.array([0.58, 0.00, 0.83]),
    'blue violet': np.array([0.54, 0.17, 0.89]),
    'BlueViolet': np.array([0.54, 0.17, 0.89]),
    'purple': np.array([0.63, 0.13, 0.94]),
    'medium purple': np.array([0.58, 0.44, 0.86]),
    'MediumPurple': np.array([0.58, 0.44, 0.86]),
    'thistle': np.array([0.85, 0.75, 0.85]),
    'snow1': np.array([1.00, 0.98, 0.98]),
    'snow2': np.array([0.93, 0.91, 0.91]),
    'snow3': np.array([0.80, 0.79, 0.79]),
    'snow4': np.array([0.55, 0.54, 0.54]),
    'seashell1': np.array([1.00, 0.96, 0.93]),
    'seashell2': np.array([0.93, 0.90, 0.87]),
    'seashell3': np.array([0.80, 0.77, 0.75]),
    'seashell4': np.array([0.55, 0.53, 0.51]),
    'AntiqueWhite1': np.array([1.00, 0.94, 0.86]),
    'AntiqueWhite2': np.array([0.93, 0.87, 0.80]),
    'AntiqueWhite3': np.array([0.80, 0.75, 0.69]),
    'AntiqueWhite4': np.array([0.55, 0.51, 0.47]),
    'bisque1': np.array([1.00, 0.89, 0.77]),
    'bisque2': np.array([0.93, 0.84, 0.72]),
    'bisque3': np.array([0.80, 0.72, 0.62]),
    'bisque4': np.array([0.55, 0.49, 0.42]),
    'PeachPuff1': np.array([1.00, 0.85, 0.73]),
    'PeachPuff2': np.array([0.93, 0.80, 0.68]),
    'PeachPuff3': np.array([0.80, 0.69, 0.58]),
    'PeachPuff4': np.array([0.55, 0.47, 0.40]),
    'NavajoWhite1': np.array([1.00, 0.87, 0.68]),
    'NavajoWhite2': np.array([0.93, 0.81, 0.63]),
    'NavajoWhite3': np.array([0.80, 0.70, 0.55]),
    'NavajoWhite4': np.array([0.55, 0.47, 0.37]),
    'LemonChiffon1': np.array([1.00, 0.98, 0.80]),
    'LemonChiffon2': np.array([0.93, 0.91, 0.75]),
    'LemonChiffon3': np.array([0.80, 0.79, 0.65]),
    'LemonChiffon4': np.array([0.55, 0.54, 0.44]),
    'cornsilk1': np.array([1.00, 0.97, 0.86]),
    'cornsilk2': np.array([0.93, 0.91, 0.80]),
    'cornsilk3': np.array([0.80, 0.78, 0.69]),
    'cornsilk4': np.array([0.55, 0.53, 0.47]),
    'ivory1': np.array([1.00, 1.00, 0.94]),
    'ivory2': np.array([0.93, 0.93, 0.88]),
    'ivory3': np.array([0.80, 0.80, 0.76]),
    'ivory4': np.array([0.55, 0.55, 0.51]),
    'honeydew1': np.array([0.94, 1.00, 0.94]),
    'honeydew2': np.array([0.88, 0.93, 0.88]),
    'honeydew3': np.array([0.76, 0.80, 0.76]),
    'honeydew4': np.array([0.51, 0.55, 0.51]),
    'LavenderBlush1': np.array([1.00, 0.94, 0.96]),
    'LavenderBlush2': np.array([0.93, 0.88, 0.90]),
    'LavenderBlush3': np.array([0.80, 0.76, 0.77]),
    'LavenderBlush4': np.array([0.55, 0.51, 0.53]),
    'MistyRose1': np.array([1.00, 0.89, 0.88]),
    'MistyRose2': np.array([0.93, 0.84, 0.82]),
    'MistyRose3': np.array([0.80, 0.72, 0.71]),
    'MistyRose4': np.array([0.55, 0.49, 0.48]),
    'azure1': np.array([0.94, 1.00, 1.00]),
    'azure2': np.array([0.88, 0.93, 0.93]),
    'azure3': np.array([0.76, 0.80, 0.80]),
    'azure4': np.array([0.51, 0.55, 0.55]),
    'SlateBlue1': np.array([0.51, 0.44, 1.00]),
    'SlateBlue2': np.array([0.48, 0.40, 0.93]),
    'SlateBlue3': np.array([0.41, 0.35, 0.80]),
    'SlateBlue4': np.array([0.28, 0.24, 0.55]),
    'RoyalBlue1': np.array([0.28, 0.46, 1.00]),
    'RoyalBlue2': np.array([0.26, 0.43, 0.93]),
    'RoyalBlue3': np.array([0.23, 0.37, 0.80]),
    'RoyalBlue4': np.array([0.15, 0.25, 0.55]),
    'blue1': np.array([0.00, 0.00, 1.00]),
    'blue2': np.array([0.00, 0.00, 0.93]),
    'blue3': np.array([0.00, 0.00, 0.80]),
    'blue4': np.array([0.00, 0.00, 0.55]),
    'DodgerBlue1': np.array([0.12, 0.56, 1.00]),
    'DodgerBlue2': np.array([0.11, 0.53, 0.93]),
    'DodgerBlue3': np.array([0.09, 0.45, 0.80]),
    'DodgerBlue4': np.array([0.06, 0.31, 0.55]),
    'SteelBlue1': np.array([0.39, 0.72, 1.00]),
    'SteelBlue2': np.array([0.36, 0.67, 0.93]),
    'SteelBlue3': np.array([0.31, 0.58, 0.80]),
    'SteelBlue4': np.array([0.21, 0.39, 0.55]),
    'DeepSkyBlue1': np.array([0.00, 0.75, 1.00]),
    'DeepSkyBlue2': np.array([0.00, 0.70, 0.93]),
    'DeepSkyBlue3': np.array([0.00, 0.60, 0.80]),
    'DeepSkyBlue4': np.array([0.00, 0.41, 0.55]),
    'SkyBlue1': np.array([0.53, 0.81, 1.00]),
    'SkyBlue2': np.array([0.49, 0.75, 0.93]),
    'SkyBlue3': np.array([0.42, 0.65, 0.80]),
    'SkyBlue4': np.array([0.29, 0.44, 0.55]),
    'LightSkyBlue1': np.array([0.69, 0.89, 1.00]),
    'LightSkyBlue2': np.array([0.64, 0.83, 0.93]),
    'LightSkyBlue3': np.array([0.55, 0.71, 0.80]),
    'LightSkyBlue4': np.array([0.38, 0.48, 0.55]),
    'SlateGray1': np.array([0.78, 0.89, 1.00]),
    'SlateGray2': np.array([0.73, 0.83, 0.93]),
    'SlateGray3': np.array([0.62, 0.71, 0.80]),
    'SlateGray4': np.array([0.42, 0.48, 0.55]),
    'LightSteelBlue1': np.array([0.79, 0.88, 1.00]),
    'LightSteelBlue2': np.array([0.74, 0.82, 0.93]),
    'LightSteelBlue3': np.array([0.64, 0.71, 0.80]),
    'LightSteelBlue4': np.array([0.43, 0.48, 0.55]),
    'LightBlue1': np.array([0.75, 0.94, 1.00]),
    'LightBlue2': np.array([0.70, 0.87, 0.93]),
    'LightBlue3': np.array([0.60, 0.75, 0.80]),
    'LightBlue4': np.array([0.41, 0.51, 0.55]),
    'LightCyan1': np.array([0.88, 1.00, 1.00]),
    'LightCyan2': np.array([0.82, 0.93, 0.93]),
    'LightCyan3': np.array([0.71, 0.80, 0.80]),
    'LightCyan4': np.array([0.48, 0.55, 0.55]),
    'PaleTurquoise1': np.array([0.73, 1.00, 1.00]),
    'PaleTurquoise2': np.array([0.68, 0.93, 0.93]),
    'PaleTurquoise3': np.array([0.59, 0.80, 0.80]),
    'PaleTurquoise4': np.array([0.40, 0.55, 0.55]),
    'CadetBlue1': np.array([0.60, 0.96, 1.00]),
    'CadetBlue2': np.array([0.56, 0.90, 0.93]),
    'CadetBlue3': np.array([0.48, 0.77, 0.80]),
    'CadetBlue4': np.array([0.33, 0.53, 0.55]),
    'turquoise1': np.array([0.00, 0.96, 1.00]),
    'turquoise2': np.array([0.00, 0.90, 0.93]),
    'turquoise3': np.array([0.00, 0.77, 0.80]),
    'turquoise4': np.array([0.00, 0.53, 0.55]),
    'cyan1': np.array([0.00, 1.00, 1.00]),
    'cyan2': np.array([0.00, 0.93, 0.93]),
    'cyan3': np.array([0.00, 0.80, 0.80]),
    'cyan4': np.array([0.00, 0.55, 0.55]),
    'DarkSlateGray1': np.array([0.59, 1.00, 1.00]),
    'DarkSlateGray2': np.array([0.55, 0.93, 0.93]),
    'DarkSlateGray3': np.array([0.47, 0.80, 0.80]),
    'DarkSlateGray4': np.array([0.32, 0.55, 0.55]),
    'aquamarine1': np.array([0.50, 1.00, 0.83]),
    'aquamarine2': np.array([0.46, 0.93, 0.78]),
    'aquamarine3': np.array([0.40, 0.80, 0.67]),
    'aquamarine4': np.array([0.27, 0.55, 0.45]),
    'DarkSeaGreen1': np.array([0.76, 1.00, 0.76]),
    'DarkSeaGreen2': np.array([0.71, 0.93, 0.71]),
    'DarkSeaGreen3': np.array([0.61, 0.80, 0.61]),
    'DarkSeaGreen4': np.array([0.41, 0.55, 0.41]),
    'SeaGreen1': np.array([0.33, 1.00, 0.62]),
    'SeaGreen2': np.array([0.31, 0.93, 0.58]),
    'SeaGreen3': np.array([0.26, 0.80, 0.50]),
    'SeaGreen4': np.array([0.18, 0.55, 0.34]),
    'PaleGreen1': np.array([0.60, 1.00, 0.60]),
    'PaleGreen2': np.array([0.56, 0.93, 0.56]),
    'PaleGreen3': np.array([0.49, 0.80, 0.49]),
    'PaleGreen4': np.array([0.33, 0.55, 0.33]),
    'SpringGreen1': np.array([0.00, 1.00, 0.50]),
    'SpringGreen2': np.array([0.00, 0.93, 0.46]),
    'SpringGreen3': np.array([0.00, 0.80, 0.40]),
    'SpringGreen4': np.array([0.00, 0.55, 0.27]),
    'green1': np.array([0.00, 1.00, 0.00]),
    'green2': np.array([0.00, 0.93, 0.00]),
    'green3': np.array([0.00, 0.80, 0.00]),
    'green4': np.array([0.00, 0.55, 0.00]),
    'chartreuse1': np.array([0.50, 1.00, 0.00]),
    'chartreuse2': np.array([0.46, 0.93, 0.00]),
    'chartreuse3': np.array([0.40, 0.80, 0.00]),
    'chartreuse4': np.array([0.27, 0.55, 0.00]),
    'OliveDrab1': np.array([0.75, 1.00, 0.24]),
    'OliveDrab2': np.array([0.70, 0.93, 0.23]),
    'OliveDrab3': np.array([0.60, 0.80, 0.20]),
    'OliveDrab4': np.array([0.41, 0.55, 0.13]),
    'DarkOliveGreen1': np.array([0.79, 1.00, 0.44]),
    'DarkOliveGreen2': np.array([0.74, 0.93, 0.41]),
    'DarkOliveGreen3': np.array([0.64, 0.80, 0.35]),
    'DarkOliveGreen4': np.array([0.43, 0.55, 0.24]),
    'khaki1': np.array([1.00, 0.96, 0.56]),
    'khaki2': np.array([0.93, 0.90, 0.52]),
    'khaki3': np.array([0.80, 0.78, 0.45]),
    'khaki4': np.array([0.55, 0.53, 0.31]),
    'LightGoldenrod1': np.array([1.00, 0.93, 0.55]),
    'LightGoldenrod2': np.array([0.93, 0.86, 0.51]),
    'LightGoldenrod3': np.array([0.80, 0.75, 0.44]),
    'LightGoldenrod4': np.array([0.55, 0.51, 0.30]),
    'LightYellow1': np.array([1.00, 1.00, 0.88]),
    'LightYellow2': np.array([0.93, 0.93, 0.82]),
    'LightYellow3': np.array([0.80, 0.80, 0.71]),
    'LightYellow4': np.array([0.55, 0.55, 0.48]),
    'yellow1': np.array([1.00, 1.00, 0.00]),
    'yellow2': np.array([0.93, 0.93, 0.00]),
    'yellow3': np.array([0.80, 0.80, 0.00]),
    'yellow4': np.array([0.55, 0.55, 0.00]),
    'gold1': np.array([1.00, 0.84, 0.00]),
    'gold2': np.array([0.93, 0.79, 0.00]),
    'gold3': np.array([0.80, 0.68, 0.00]),
    'gold4': np.array([0.55, 0.46, 0.00]),
    'goldenrod1': np.array([1.00, 0.76, 0.15]),
    'goldenrod2': np.array([0.93, 0.71, 0.13]),
    'goldenrod3': np.array([0.80, 0.61, 0.11]),
    'goldenrod4': np.array([0.55, 0.41, 0.08]),
    'DarkGoldenrod1': np.array([1.00, 0.73, 0.06]),
    'DarkGoldenrod2': np.array([0.93, 0.68, 0.05]),
    'DarkGoldenrod3': np.array([0.80, 0.58, 0.05]),
    'DarkGoldenrod4': np.array([0.55, 0.40, 0.03]),
    'RosyBrown1': np.array([1.00, 0.76, 0.76]),
    'RosyBrown2': np.array([0.93, 0.71, 0.71]),
    'RosyBrown3': np.array([0.80, 0.61, 0.61]),
    'RosyBrown4': np.array([0.55, 0.41, 0.41]),
    'IndianRed1': np.array([1.00, 0.42, 0.42]),
    'IndianRed2': np.array([0.93, 0.39, 0.39]),
    'IndianRed3': np.array([0.80, 0.33, 0.33]),
    'IndianRed4': np.array([0.55, 0.23, 0.23]),
    'sienna1': np.array([1.00, 0.51, 0.28]),
    'sienna2': np.array([0.93, 0.47, 0.26]),
    'sienna3': np.array([0.80, 0.41, 0.22]),
    'sienna4': np.array([0.55, 0.28, 0.15]),
    'burlywood1': np.array([1.00, 0.83, 0.61]),
    'burlywood2': np.array([0.93, 0.77, 0.57]),
    'burlywood3': np.array([0.80, 0.67, 0.49]),
    'burlywood4': np.array([0.55, 0.45, 0.33]),
    'wheat1': np.array([1.00, 0.91, 0.73]),
    'wheat2': np.array([0.93, 0.85, 0.68]),
    'wheat3': np.array([0.80, 0.73, 0.59]),
    'wheat4': np.array([0.55, 0.49, 0.40]),
    'tan1': np.array([1.00, 0.65, 0.31]),
    'tan2': np.array([0.93, 0.60, 0.29]),
    'tan3': np.array([0.80, 0.52, 0.25]),
    'tan4': np.array([0.55, 0.35, 0.17]),
    'chocolate1': np.array([1.00, 0.50, 0.14]),
    'chocolate2': np.array([0.93, 0.46, 0.13]),
    'chocolate3': np.array([0.80, 0.40, 0.11]),
    'chocolate4': np.array([0.55, 0.27, 0.07]),
    'firebrick1': np.array([1.00, 0.19, 0.19]),
    'firebrick2': np.array([0.93, 0.17, 0.17]),
    'firebrick3': np.array([0.80, 0.15, 0.15]),
    'firebrick4': np.array([0.55, 0.10, 0.10]),
    'brown1': np.array([1.00, 0.25, 0.25]),
    'brown2': np.array([0.93, 0.23, 0.23]),
    'brown3': np.array([0.80, 0.20, 0.20]),
    'brown4': np.array([0.55, 0.14, 0.14]),
    'salmon1': np.array([1.00, 0.55, 0.41]),
    'salmon2': np.array([0.93, 0.51, 0.38]),
    'salmon3': np.array([0.80, 0.44, 0.33]),
    'salmon4': np.array([0.55, 0.30, 0.22]),
    'LightSalmon1': np.array([1.00, 0.63, 0.48]),
    'LightSalmon2': np.array([0.93, 0.58, 0.45]),
    'LightSalmon3': np.array([0.80, 0.51, 0.38]),
    'LightSalmon4': np.array([0.55, 0.34, 0.26]),
    'orange1': np.array([1.00, 0.65, 0.00]),
    'orange2': np.array([0.93, 0.60, 0.00]),
    'orange3': np.array([0.80, 0.52, 0.00]),
    'orange4': np.array([0.55, 0.35, 0.00]),
    'DarkOrange1': np.array([1.00, 0.50, 0.00]),
    'DarkOrange2': np.array([0.93, 0.46, 0.00]),
    'DarkOrange3': np.array([0.80, 0.40, 0.00]),
    'DarkOrange4': np.array([0.55, 0.27, 0.00]),
    'coral1': np.array([1.00, 0.45, 0.34]),
    'coral2': np.array([0.93, 0.42, 0.31]),
    'coral3': np.array([0.80, 0.36, 0.27]),
    'coral4': np.array([0.55, 0.24, 0.18]),
    'tomato1': np.array([1.00, 0.39, 0.28]),
    'tomato2': np.array([0.93, 0.36, 0.26]),
    'tomato3': np.array([0.80, 0.31, 0.22]),
    'tomato4': np.array([0.55, 0.21, 0.15]),
    'OrangeRed1': np.array([1.00, 0.27, 0.00]),
    'OrangeRed2': np.array([0.93, 0.25, 0.00]),
    'OrangeRed3': np.array([0.80, 0.22, 0.00]),
    'OrangeRed4': np.array([0.55, 0.15, 0.00]),
    'red1': np.array([1.00, 0.00, 0.00]),
    'red2': np.array([0.93, 0.00, 0.00]),
    'red3': np.array([0.80, 0.00, 0.00]),
    'red4': np.array([0.55, 0.00, 0.00]),
    'DeepPink1': np.array([1.00, 0.08, 0.58]),
    'DeepPink2': np.array([0.93, 0.07, 0.54]),
    'DeepPink3': np.array([0.80, 0.06, 0.46]),
    'DeepPink4': np.array([0.55, 0.04, 0.31]),
    'HotPink1': np.array([1.00, 0.43, 0.71]),
    'HotPink2': np.array([0.93, 0.42, 0.65]),
    'HotPink3': np.array([0.80, 0.38, 0.56]),
    'HotPink4': np.array([0.55, 0.23, 0.38]),
    'pink1': np.array([1.00, 0.71, 0.77]),
    'pink2': np.array([0.93, 0.66, 0.72]),
    'pink3': np.array([0.80, 0.57, 0.62]),
    'pink4': np.array([0.55, 0.39, 0.42]),
    'LightPink1': np.array([1.00, 0.68, 0.73]),
    'LightPink2': np.array([0.93, 0.64, 0.68]),
    'LightPink3': np.array([0.80, 0.55, 0.58]),
    'LightPink4': np.array([0.55, 0.37, 0.40]),
    'PaleVioletRed1': np.array([1.00, 0.51, 0.67]),
    'PaleVioletRed2': np.array([0.93, 0.47, 0.62]),
    'PaleVioletRed3': np.array([0.80, 0.41, 0.54]),
    'PaleVioletRed4': np.array([0.55, 0.28, 0.36]),
    'maroon1': np.array([1.00, 0.20, 0.70]),
    'maroon2': np.array([0.93, 0.19, 0.65]),
    'maroon3': np.array([0.80, 0.16, 0.56]),
    'maroon4': np.array([0.55, 0.11, 0.38]),
    'VioletRed1': np.array([1.00, 0.24, 0.59]),
    'VioletRed2': np.array([0.93, 0.23, 0.55]),
    'VioletRed3': np.array([0.80, 0.20, 0.47]),
    'VioletRed4': np.array([0.55, 0.13, 0.32]),
    'magenta1': np.array([1.00, 0.00, 1.00]),
    'magenta2': np.array([0.93, 0.00, 0.93]),
    'magenta3': np.array([0.80, 0.00, 0.80]),
    'magenta4': np.array([0.55, 0.00, 0.55]),
    'orchid1': np.array([1.00, 0.51, 0.98]),
    'orchid2': np.array([0.93, 0.48, 0.91]),
    'orchid3': np.array([0.80, 0.41, 0.79]),
    'orchid4': np.array([0.55, 0.28, 0.54]),
    'plum1': np.array([1.00, 0.73, 1.00]),
    'plum2': np.array([0.93, 0.68, 0.93]),
    'plum3': np.array([0.80, 0.59, 0.80]),
    'plum4': np.array([0.55, 0.40, 0.55]),
    'MediumOrchid1': np.array([0.88, 0.40, 1.00]),
    'MediumOrchid2': np.array([0.82, 0.37, 0.93]),
    'MediumOrchid3': np.array([0.71, 0.32, 0.80]),
    'MediumOrchid4': np.array([0.48, 0.22, 0.55]),
    'DarkOrchid1': np.array([0.75, 0.24, 1.00]),
    'DarkOrchid2': np.array([0.70, 0.23, 0.93]),
    'DarkOrchid3': np.array([0.60, 0.20, 0.80]),
    'DarkOrchid4': np.array([0.41, 0.13, 0.55]),
    'purple1': np.array([0.61, 0.19, 1.00]),
    'purple2': np.array([0.57, 0.17, 0.93]),
    'purple3': np.array([0.49, 0.15, 0.80]),
    'purple4': np.array([0.33, 0.10, 0.55]),
    'MediumPurple1': np.array([0.67, 0.51, 1.00]),
    'MediumPurple2': np.array([0.62, 0.47, 0.93]),
    'MediumPurple3': np.array([0.54, 0.41, 0.80]),
    'MediumPurple4': np.array([0.36, 0.28, 0.55]),
    'thistle1': np.array([1.00, 0.88, 1.00]),
    'thistle2': np.array([0.93, 0.82, 0.93]),
    'thistle3': np.array([0.80, 0.71, 0.80]),
    'thistle4': np.array([0.55, 0.48, 0.55]),
    'gray0': np.array([0.00, 0.00, 0.00]),
    'grey0': np.array([0.00, 0.00, 0.00]),
    'gray1': np.array([0.01, 0.01, 0.01]),
    'grey1': np.array([0.01, 0.01, 0.01]),
    'gray2': np.array([0.02, 0.02, 0.02]),
    'grey2': np.array([0.02, 0.02, 0.02]),
    'gray3': np.array([0.03, 0.03, 0.03]),
    'grey3': np.array([0.03, 0.03, 0.03]),
    'gray4': np.array([0.04, 0.04, 0.04]),
    'grey4': np.array([0.04, 0.04, 0.04]),
    'gray5': np.array([0.05, 0.05, 0.05]),
    'grey5': np.array([0.05, 0.05, 0.05]),
    'gray6': np.array([0.06, 0.06, 0.06]),
    'grey6': np.array([0.06, 0.06, 0.06]),
    'gray7': np.array([0.07, 0.07, 0.07]),
    'grey7': np.array([0.07, 0.07, 0.07]),
    'gray8': np.array([0.08, 0.08, 0.08]),
    'grey8': np.array([0.08, 0.08, 0.08]),
    'gray9': np.array([0.09, 0.09, 0.09]),
    'grey9': np.array([0.09, 0.09, 0.09]),
    'gray10': np.array([0.10, 0.10, 0.10]),
    'grey10': np.array([0.10, 0.10, 0.10]),
    'gray11': np.array([0.11, 0.11, 0.11]),
    'grey11': np.array([0.11, 0.11, 0.11]),
    'gray12': np.array([0.12, 0.12, 0.12]),
    'grey12': np.array([0.12, 0.12, 0.12]),
    'gray13': np.array([0.13, 0.13, 0.13]),
    'grey13': np.array([0.13, 0.13, 0.13]),
    'gray14': np.array([0.14, 0.14, 0.14]),
    'grey14': np.array([0.14, 0.14, 0.14]),
    'gray15': np.array([0.15, 0.15, 0.15]),
    'grey15': np.array([0.15, 0.15, 0.15]),
    'gray16': np.array([0.16, 0.16, 0.16]),
    'grey16': np.array([0.16, 0.16, 0.16]),
    'gray17': np.array([0.17, 0.17, 0.17]),
    'grey17': np.array([0.17, 0.17, 0.17]),
    'gray18': np.array([0.18, 0.18, 0.18]),
    'grey18': np.array([0.18, 0.18, 0.18]),
    'gray19': np.array([0.19, 0.19, 0.19]),
    'grey19': np.array([0.19, 0.19, 0.19]),
    'gray20': np.array([0.20, 0.20, 0.20]),
    'grey20': np.array([0.20, 0.20, 0.20]),
    'gray21': np.array([0.21, 0.21, 0.21]),
    'grey21': np.array([0.21, 0.21, 0.21]),
    'gray22': np.array([0.22, 0.22, 0.22]),
    'grey22': np.array([0.22, 0.22, 0.22]),
    'gray23': np.array([0.23, 0.23, 0.23]),
    'grey23': np.array([0.23, 0.23, 0.23]),
    'gray24': np.array([0.24, 0.24, 0.24]),
    'grey24': np.array([0.24, 0.24, 0.24]),
    'gray25': np.array([0.25, 0.25, 0.25]),
    'grey25': np.array([0.25, 0.25, 0.25]),
    'gray26': np.array([0.26, 0.26, 0.26]),
    'grey26': np.array([0.26, 0.26, 0.26]),
    'gray27': np.array([0.27, 0.27, 0.27]),
    'grey27': np.array([0.27, 0.27, 0.27]),
    'gray28': np.array([0.28, 0.28, 0.28]),
    'grey28': np.array([0.28, 0.28, 0.28]),
    'gray29': np.array([0.29, 0.29, 0.29]),
    'grey29': np.array([0.29, 0.29, 0.29]),
    'gray30': np.array([0.30, 0.30, 0.30]),
    'grey30': np.array([0.30, 0.30, 0.30]),
    'gray31': np.array([0.31, 0.31, 0.31]),
    'grey31': np.array([0.31, 0.31, 0.31]),
    'gray32': np.array([0.32, 0.32, 0.32]),
    'grey32': np.array([0.32, 0.32, 0.32]),
    'gray33': np.array([0.33, 0.33, 0.33]),
    'grey33': np.array([0.33, 0.33, 0.33]),
    'gray34': np.array([0.34, 0.34, 0.34]),
    'grey34': np.array([0.34, 0.34, 0.34]),
    'gray35': np.array([0.35, 0.35, 0.35]),
    'grey35': np.array([0.35, 0.35, 0.35]),
    'gray36': np.array([0.36, 0.36, 0.36]),
    'grey36': np.array([0.36, 0.36, 0.36]),
    'gray37': np.array([0.37, 0.37, 0.37]),
    'grey37': np.array([0.37, 0.37, 0.37]),
    'gray38': np.array([0.38, 0.38, 0.38]),
    'grey38': np.array([0.38, 0.38, 0.38]),
    'gray39': np.array([0.39, 0.39, 0.39]),
    'grey39': np.array([0.39, 0.39, 0.39]),
    'gray40': np.array([0.40, 0.40, 0.40]),
    'grey40': np.array([0.40, 0.40, 0.40]),
    'gray41': np.array([0.41, 0.41, 0.41]),
    'grey41': np.array([0.41, 0.41, 0.41]),
    'gray42': np.array([0.42, 0.42, 0.42]),
    'grey42': np.array([0.42, 0.42, 0.42]),
    'gray43': np.array([0.43, 0.43, 0.43]),
    'grey43': np.array([0.43, 0.43, 0.43]),
    'gray44': np.array([0.44, 0.44, 0.44]),
    'grey44': np.array([0.44, 0.44, 0.44]),
    'gray45': np.array([0.45, 0.45, 0.45]),
    'grey45': np.array([0.45, 0.45, 0.45]),
    'gray46': np.array([0.46, 0.46, 0.46]),
    'grey46': np.array([0.46, 0.46, 0.46]),
    'gray47': np.array([0.47, 0.47, 0.47]),
    'grey47': np.array([0.47, 0.47, 0.47]),
    'gray48': np.array([0.48, 0.48, 0.48]),
    'grey48': np.array([0.48, 0.48, 0.48]),
    'gray49': np.array([0.49, 0.49, 0.49]),
    'grey49': np.array([0.49, 0.49, 0.49]),
    'gray50': np.array([0.50, 0.50, 0.50]),
    'grey50': np.array([0.50, 0.50, 0.50]),
    'gray51': np.array([0.51, 0.51, 0.51]),
    'grey51': np.array([0.51, 0.51, 0.51]),
    'gray52': np.array([0.52, 0.52, 0.52]),
    'grey52': np.array([0.52, 0.52, 0.52]),
    'gray53': np.array([0.53, 0.53, 0.53]),
    'grey53': np.array([0.53, 0.53, 0.53]),
    'gray54': np.array([0.54, 0.54, 0.54]),
    'grey54': np.array([0.54, 0.54, 0.54]),
    'gray55': np.array([0.55, 0.55, 0.55]),
    'grey55': np.array([0.55, 0.55, 0.55]),
    'gray56': np.array([0.56, 0.56, 0.56]),
    'grey56': np.array([0.56, 0.56, 0.56]),
    'gray57': np.array([0.57, 0.57, 0.57]),
    'grey57': np.array([0.57, 0.57, 0.57]),
    'gray58': np.array([0.58, 0.58, 0.58]),
    'grey58': np.array([0.58, 0.58, 0.58]),
    'gray59': np.array([0.59, 0.59, 0.59]),
    'grey59': np.array([0.59, 0.59, 0.59]),
    'gray60': np.array([0.60, 0.60, 0.60]),
    'grey60': np.array([0.60, 0.60, 0.60]),
    'gray61': np.array([0.61, 0.61, 0.61]),
    'grey61': np.array([0.61, 0.61, 0.61]),
    'gray62': np.array([0.62, 0.62, 0.62]),
    'grey62': np.array([0.62, 0.62, 0.62]),
    'gray63': np.array([0.63, 0.63, 0.63]),
    'grey63': np.array([0.63, 0.63, 0.63]),
    'gray64': np.array([0.64, 0.64, 0.64]),
    'grey64': np.array([0.64, 0.64, 0.64]),
    'gray65': np.array([0.65, 0.65, 0.65]),
    'grey65': np.array([0.65, 0.65, 0.65]),
    'gray66': np.array([0.66, 0.66, 0.66]),
    'grey66': np.array([0.66, 0.66, 0.66]),
    'gray67': np.array([0.67, 0.67, 0.67]),
    'grey67': np.array([0.67, 0.67, 0.67]),
    'gray68': np.array([0.68, 0.68, 0.68]),
    'grey68': np.array([0.68, 0.68, 0.68]),
    'gray69': np.array([0.69, 0.69, 0.69]),
    'grey69': np.array([0.69, 0.69, 0.69]),
    'gray70': np.array([0.70, 0.70, 0.70]),
    'grey70': np.array([0.70, 0.70, 0.70]),
    'gray71': np.array([0.71, 0.71, 0.71]),
    'grey71': np.array([0.71, 0.71, 0.71]),
    'gray72': np.array([0.72, 0.72, 0.72]),
    'grey72': np.array([0.72, 0.72, 0.72]),
    'gray73': np.array([0.73, 0.73, 0.73]),
    'grey73': np.array([0.73, 0.73, 0.73]),
    'gray74': np.array([0.74, 0.74, 0.74]),
    'grey74': np.array([0.74, 0.74, 0.74]),
    'gray75': np.array([0.75, 0.75, 0.75]),
    'grey75': np.array([0.75, 0.75, 0.75]),
    'gray76': np.array([0.76, 0.76, 0.76]),
    'grey76': np.array([0.76, 0.76, 0.76]),
    'gray77': np.array([0.77, 0.77, 0.77]),
    'grey77': np.array([0.77, 0.77, 0.77]),
    'gray78': np.array([0.78, 0.78, 0.78]),
    'grey78': np.array([0.78, 0.78, 0.78]),
    'gray79': np.array([0.79, 0.79, 0.79]),
    'grey79': np.array([0.79, 0.79, 0.79]),
    'gray80': np.array([0.80, 0.80, 0.80]),
    'grey80': np.array([0.80, 0.80, 0.80]),
    'gray81': np.array([0.81, 0.81, 0.81]),
    'grey81': np.array([0.81, 0.81, 0.81]),
    'gray82': np.array([0.82, 0.82, 0.82]),
    'grey82': np.array([0.82, 0.82, 0.82]),
    'gray83': np.array([0.83, 0.83, 0.83]),
    'grey83': np.array([0.83, 0.83, 0.83]),
    'gray84': np.array([0.84, 0.84, 0.84]),
    'grey84': np.array([0.84, 0.84, 0.84]),
    'gray85': np.array([0.85, 0.85, 0.85]),
    'grey85': np.array([0.85, 0.85, 0.85]),
    'gray86': np.array([0.86, 0.86, 0.86]),
    'grey86': np.array([0.86, 0.86, 0.86]),
    'gray87': np.array([0.87, 0.87, 0.87]),
    'grey87': np.array([0.87, 0.87, 0.87]),
    'gray88': np.array([0.88, 0.88, 0.88]),
    'grey88': np.array([0.88, 0.88, 0.88]),
    'gray89': np.array([0.89, 0.89, 0.89]),
    'grey89': np.array([0.89, 0.89, 0.89]),
    'gray90': np.array([0.90, 0.90, 0.90]),
    'grey90': np.array([0.90, 0.90, 0.90]),
    'gray91': np.array([0.91, 0.91, 0.91]),
    'grey91': np.array([0.91, 0.91, 0.91]),
    'gray92': np.array([0.92, 0.92, 0.92]),
    'grey92': np.array([0.92, 0.92, 0.92]),
    'gray93': np.array([0.93, 0.93, 0.93]),
    'grey93': np.array([0.93, 0.93, 0.93]),
    'gray94': np.array([0.94, 0.94, 0.94]),
    'grey94': np.array([0.94, 0.94, 0.94]),
    'gray95': np.array([0.95, 0.95, 0.95]),
    'grey95': np.array([0.95, 0.95, 0.95]),
    'gray96': np.array([0.96, 0.96, 0.96]),
    'grey96': np.array([0.96, 0.96, 0.96]),
    'gray97': np.array([0.97, 0.97, 0.97]),
    'grey97': np.array([0.97, 0.97, 0.97]),
    'gray98': np.array([0.98, 0.98, 0.98]),
    'grey98': np.array([0.98, 0.98, 0.98]),
    'gray99': np.array([0.99, 0.99, 0.99]),
    'grey99': np.array([0.99, 0.99, 0.99]),
    'gray100': np.array([1.00, 1.00, 1.00]),
    'grey100': np.array([1.00, 1.00, 1.00]),
    'dark grey': np.array([0.66, 0.66, 0.66]),
    'DarkGrey': np.array([0.66, 0.66, 0.66]),
    'dark gray': np.array([0.66, 0.66, 0.66]),
    'DarkGray': np.array([0.66, 0.66, 0.66]),
    'dark blue': np.array([0.00, 0.00, 0.55]),
    'DarkBlue': np.array([0.00, 0.00, 0.55]),
    'dark cyan': np.array([0.00, 0.55, 0.55]),
    'DarkCyan': np.array([0.00, 0.55, 0.55]),
    'dark magenta': np.array([0.55, 0.00, 0.55]),
    'DarkMagenta': np.array([0.55, 0.00, 0.55]),
    'dark red': np.array([0.55, 0.00, 0.00]),
    'DarkRed': np.array([0.55, 0.00, 0.00]),
    'light green': np.array([0.56, 0.93, 0.56]),
    'LightGreen': np.array([0.56, 0.93, 0.56])
}

# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.01.02

import numpy as np
import cv2
import os
import trimesh



colors = {
    'pink': [.6, .0, .4],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .2, .1],
    'brown-light': [0.654, 0.396, 0.164],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [1., .2, 0],

    'grey': [.7, .7, .7],
    'grey-blue': [0.345, 0.580, 0.713],
    'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83, 1, 0],
}

def imagearray2file(img_array, outpath=None, fps=30):
    '''
    :param nparray: RxCxTxwidthxheightx3
    :param outpath: the directory where T images will be dumped for each time point in range T
    :param fps: fps of the gif file
    :return:
        it will return an image list with length T
        if outpath is given as a png file, an image will be saved for each t in T.
        if outpath is given as a gif file, an animated image with T frames will be created.
    '''

    if outpath is not None:
        outdir = os.path.dirname(outpath)
        if not os.path.exists(outdir): os.makedirs(outdir)

    if not isinstance(img_array, np.ndarray) or img_array.ndim < 6:
        raise ValueError('img_array should be a numpy array of shape RxCxTxwidthxheightx3')

    R, C, T, img_h, img_w, img_c = img_array.shape

    out_images = []
    for tIdx in range(T):
        row_images = []
        for rIdx in range(R):
            col_images = []
            for cIdx in range(C):
                col_images.append(img_array[rIdx, cIdx, tIdx])
            row_images.append(np.hstack(col_images))
        t_image = np.vstack(row_images)
        out_images.append(t_image)

    if outpath is not None:
        ext = outpath.split('.')[-1]
        if ext in ['png', 'jpeg', 'jpg']:
            for tIdx in range(T):
                if T > 1:
                    cur_outpath = outpath.replace('.%s'%ext, '_%03d.%s'%(tIdx, ext))
                else:
                    cur_outpath = outpath
                    
                img = cv2.cvtColor(out_images[tIdx], cv2.COLOR_BGR2RGB)
                cv2.imwrite(cur_outpath, img)
                while not os.path.exists(cur_outpath): continue  # wait until the snapshot is written to the disk
        elif ext == 'gif':
            import imageio
            with imageio.get_writer(outpath, mode='I', fps = fps) as writer:
                for tIdx in range(T):
                    img = out_images[tIdx].astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    writer.append_data(img)
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(outpath, fourcc, fps, (img_w, img_h), True)
            for tIdx in range(T):
                img = out_images[tIdx].astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)

            video.release()
            cv2.destroyAllWindows()
        elif ext == 'mp4':
            #
            # from moviepy.editor import ImageSequenceClip
            # animation = ImageSequenceClip(out_images, fps=fps)
            # animation.write_videofile(outpath, verbose=False)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(outpath, fourcc, fps, (img_w, img_h), True)
            for tIdx in range(T):
                img = out_images[tIdx].astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)

            video.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass

    return out_images

def render_smpl_params(bm, body_parms):
    '''
    :param bm: pytorch body model with batch_size 1
    :param pose_body: Nx21x3
    :param trans: Nx3
    :param betas: Nxnum_betas
    :return: N x 400 x 400 x 3
    '''

    from human_body_prior.tools.omni_tools import copy2cpu as c2c
    from body_visualizer.mesh.mesh_viewer import MeshViewer

    imw, imh = 400, 400

    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    faces = c2c(bm.f)

    v = c2c(bm(**body_parms).v)

    T, num_verts = v.shape[:-1]

    images = []
    for fIdx in range(T):

        mesh = trimesh.base.Trimesh(v[fIdx], faces, vertex_colors=num_verts*colors['grey'])
        mv.set_meshes([mesh], 'static')

        images.append(mv.render())

    return np.array(images).reshape(T, imw, imh, 3)

def meshes_as_png(meshes, outpath=None, view_angles=[0, 180]):
    from body_visualizer.mesh.mesh_viewer import MeshViewer

    imw = 800
    imh = 800
    mv = MeshViewer(imh, imw)
    mv.set_cam_trans([0, -.5, 1.75])
    images = np.zeros([len(meshes), len(view_angles), 1, imw, imh, 3])
    for mIdx, mesh in enumerate(meshes):
        for rId, angle in enumerate(view_angles):
            if angle != 0: mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
            mv.set_meshes([mesh], group_name='static')
            images[mIdx, rId, 0] = cv2.cvtColor(mv.render(render_wireframe=False), cv2.COLOR_BGR2RGB)
            if angle != 0: mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

    if outpath is not None: imagearray2file(images, outpath)
    return images

def show_image(img_ndarray):
    '''
    Visualize rendered body images resulted from render_smpl_params in Jupyter notebook
    :param img_ndarray: Nxim_hxim_wx3
    '''
    import matplotlib.pyplot as plt
    import cv2
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis('off')

    # fig.canvas.draw()
    # return True
if __name__ == '__main__':
    main()
