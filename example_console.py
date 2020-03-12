"""
Example of how to use videofig from the console.
You can load this into your ipython console via: %load example_console.py
"""

import os
import sys
from matplotlib import pyplot as plt
from PIL import Image
from skimage.io import imread_collection

from videofig import videofig, redraw_fn

# This is the function that decides what will be drawn for each index f.
# Note that this is completely generalizable.  Here we are just stepping through a
# collection of images from skimage.  But this could also be used to step
# through a dataframe containing a train or test split of the data, for example.
def my_draw(f, seq):
    img = seq[f]
    fullpath = seq._files[f]
    title = "{} - {} w x {} h".format(os.path.basename(fullpath), img.shape[1], img.shape[0])
    return (img, None, title)

def overlay_annots(ax, f, seq):
    pass

def drawcoll(seq):
    f = videofig(len(seq), redraw_fn, proc_func=lambda f: (my_draw(f, seq)),
         overlay_func = lambda ax,f: (overlay_annots(ax,f,seq)), play_fps=2)

# Example of how to have multiple videofigs at once

# First 10K images
drawcoll(imread_collection("/Users/lambertw/images/places365-std/val_256/Places365_val_0000*.jpg",
                        conserve_memory=True))

# If you're running this all via %load, then the second window might appear directly on top of the
# first window so you may have to move it to see the second.

# Second 10K images
drawcoll(imread_collection("/Users/lambertw/images/places365-std/val_256/Places365_val_0001*.jpg",
                        conserve_memory=True))

