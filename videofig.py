#! /usr/bin/env python
# -*- coding: utf-8 -*-
# $Id: videofig.py 613 2017-08-11 22:46:35Z lambertw $
#
# Copyright 2017 Lambert Wixson, based on MIT-licensed and 
# copyrighted work from © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license, as follows:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Figure with horizontal scrollbar and play capabilities

For latest version, go to https://github.com/lambertwx/videofig

Basic usage
-----------
Creates a figure with a horizontal scrollbar and shortcuts to scroll automatically.
The scroll range is 0 to NUM_FRAMES - 1. The function REDRAW_FUN(F, AXES) is called to
redraw at scroll position F (for example, REDRAW_FUNC can show the frame F of a video)
using AXES for drawing. F is an integer, AXES is a instance of [Axes class](https://matplotlib.org/api/axes_api.html)

This can be used not only to play and analyze standard videos, but it also lets you place
any custom Matplotlib plots and graphics on top.

The keyboard shortcuts are:
  Enter(Return) -- play/pause video (25 frames-per-second default).
  Backspace -- play/pause video 5 times slower.
  Right/left arrow keys -- advance/go back one frame.
  Page down/page up -- advance/go back 30 frames.
  Home/end -- go to first/last frame of video.

Advanced usage
--------------
videofig(NUM_FRAMES, REDRAW_FUNC, FPS, BIG_SCROLL)
Also specifies the speed of the play function (frames-per-second) and
the frame step of page up/page down (or empty for defaults).

videofig(NUM_FRAMES, REDRAW_FUNC, FPS, BIG_SCROLL, KEY_FUNC)
Also calls KEY_FUNC(KEY) with any keys that weren't processed, so you
can add more shortcut keys (or empty for none).

Example 1: Plot a dynamic sine wave
---------
  import numpy as np

  def redraw_fn(f, fig, axes, proc_func, cmap):
    amp = float(f) / 3000
    f0 = 3
    t = np.arange(0.0, 1.0, 0.001)
    s = amp * np.sin(2 * np.pi * f0 * t)
    if not hasattr(fig, 'redraw_fn'):
      fig.redraw_fn = lambda: None
      fig.redraw_fn.initialized = False
    if not fig.redraw_fn.initialized:
      fig.redraw_fn.l, = axes.plot(t, s, lw=2, color='red')
      fig.redraw_fn.initialized = True
    else:
      fig.redraw_fn.l.set_ydata(s)

  redraw_fn.initialized = False

  videofig(100, redraw_fn)
  
Example 2: Show images in a custom directory
---------
  import os
  import glob
  from scipy.misc import imread

  img_dir = 'YOUR-IMAGE-DIRECTORY'
  img_files = glob.glob(os.path.join(video_dir, '*.jpg'))

  def redraw_fn(f, fig, axes, proc_func, cmap):
    img_file = img_files[f]
    img = imread(img_file)
    if not hasattr(fig, 'redraw_fn'):
      fig.redraw_fn = lambda: None
      fig.redraw_fn.initialized = False
    if not fig.redraw_fn.initialized:
      fig.redraw_fn.im = axes.imshow(img, animated=True)
      fig.redraw_fn.initialized = True
    else:
      fig.redraw_fn.im.set_array(img)
  redraw_fn.initialized = False

  videofig(len(img_files), redraw_fn, play_fps=30)

Example 3: Show images together with object bounding boxes
----------
  import os
  import glob
  from scipy.misc import imread
  from matplotlib.pyplot import Rectangle
  
  video_dir = 'YOUR-VIDEO-DIRECTORY'

  img_files = glob.glob(os.path.join(video_dir, '*.jpg'))
  box_files = glob.glob(os.path.join(video_dir, '*.txt'))

  def redraw_fn(f, fig, axes, proc_func, cmap):
    img = imread(img_files[f])
    box = bbread(box_files[f])  # Define your own bounding box reading utility
    x, y, w, h = box
    if not hasattr(fig, 'redraw_fn'):
      fig.redraw_fn = lambda: None
      fig.redraw_fn.initialized = False
    if not fig.redraw_fn.initialized:
      im = axes.imshow(img, animated=True)
      bb = Rectangle((x, y), w, h,
                     fill=False,  # remove background
                     edgecolor="red")
      axes.add_patch(bb)
      fig.redraw_fn.im = im
      fig.redraw_fn.bb = bb
      fig.redraw_fn.initialized = True
    else:
      fig.redraw_fn.im.set_array(img)
      fig.redraw_fn.bb.set_xy((x, y))
      fig.redraw_fn.bb.set_width(w)
      fig.redraw_fn.bb.set_height(h)
  redraw_fn.initialized = False

  videofig(len(img_files), redraw_fn, play_fps=30)
  
Example 4: Apply horizontal Sobel filter to a scikit-image image sequence
----------
  import os
  import skimage
  from skimage import color, io, filters
  
  video_dir = 'YOUR-VIDEO-DIRECTORY'
  seq = io.imread_collection(os.path.join(video_dir, 'img*.png'), conserve_memory=True) 
  
  # The calls below use the default redraw_fn, which calls proc_func.
  
  # Display the raw images
  videofig(len(seq), redraw_fn, play_fps=30, 
           proc_func=lambda f: seq[f] )  
  
  # Display the filtered images.  We return a 2-tuple from proc_func.  The second element
  # could be a list of regions, which would be displayed by the draw_regions() function in videofig.py
  videofig(len(seq), redraw_fn, play_fps=30, 
         proc_func=lambda f: (filters.sobel_h(color.rgb2gray(seq[f])), None),
         cmap='viridis')
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
import matplotlib.patches
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

#%%
def videofig(num_frames, redraw_func, play_fps=25, big_scroll=30, key_func=None, proc_func = None, 
             fig = None, cmap = None, winname=None, overlay_func = None, vmin=None, vmax=None, *args):
  """Figure with horizontal scrollbar and play capabilities
  
  This script is mainly inspired by the elegant work of João Filipe Henriques
    https://www.mathworks.com/matlabcentral/fileexchange/29544-figure-to-play-and-analyze-videos-with-custom-plots-on-top?focused=5172704&tab=function
    
  :param num_frames: an integer, number of frames in a sequence
  :param redraw_func: callable with signature redraw_func(f, axes)
                      used to draw a new frame at position f using axes, which is a instance of Axes class in matplotlib 
  :param play_fps: an integer, number of frames per second, used to control the play speed
  :param big_scroll: an integer, big scroll number used when pressed page down or page up keys. 
  :param key_func: optional callable which signature key_func(key), used to provide custom key shortcuts.
  
  :param proc_func: The processing function that will be called by redraw_func.  Must return a 2-tuple (image, regions).
  
  :param fig: An already-created matplotlib figure to use.  If this parameter is omitted, a new figure will be created.

  :param cmap: A colormap to use.  If you are trying to display floating-point numbers, I suggest
               you pass in 'viridis'
  
  :param winname: A title for your matplotlib window.

  :param args: other optional arguments
  :return: The figure that was created (or was passed in)
  """
  # Check arguments
  check_int_scalar(num_frames, 'num_frames')
  check_callback(redraw_func, 'redraw_func')
  check_int_scalar(play_fps, 'play_fps')
  check_int_scalar(big_scroll, 'big_scroll')
  if key_func:
    check_callback(key_func, 'key_func')
  if proc_func:
      check_callback(proc_func, 'proc_func')
  if overlay_func:
      check_callback(overlay_func, 'overlay_func')
      
  # Initialize figure
  if fig:
      fig_handle = fig
  else:
      fig_handle = plt.figure()
  if winname:
        fig_handle.canvas.set_window_title(winname)
        
  # main drawing axes for video display
  if 1:
      axes_handle = fig_handle.add_axes([0, 0.03, 1, 0.97])
      axes_handle.set_axis_off()
  else:
      axes_handle = fig_handle.gca()

  # Build scrollbar
  scroll_axes_handle = fig_handle.add_axes([0, 0, 1, 0.03], facecolor='lightgoldenrodyellow')
  scroll_handle = Slider(scroll_axes_handle, '', 0.0, num_frames - 1, valinit=0.0)

  def draw_new(_):
    # Set to the right axes and call the custom redraw function
    fig_handle.sca(axes_handle)
    redraw_func(int(scroll_handle.val), fig_handle, axes_handle, proc_func, cmap, overlay_func=overlay_func, vmin=vmin, vmax=vmax)
    fig_handle.canvas.draw_idle()
    #print("In draw_new()")

  def scroll(new_f):
    new_f = min(max(new_f, 0), num_frames - 1)  # clip in the range of [0, num_frames - 1]
    cur_f = scroll_handle.val

    # Stop player at the end of the sequence
    if new_f == (num_frames - 1):
      play.running = False

    if cur_f != new_f:
      # move scroll bar to new position.  The set_val results in a call to the 
      # scroll_handle's onchanged handler, which is our draw_new()
      scroll_handle.set_val(new_f)
    #print("In scroll()")
    return axes_handle

  def play(period):
    play.running ^= True  # Toggle state
    if play.running:
      frame_idxs = range(int(scroll_handle.val), num_frames)
      play.anim = FuncAnimation(fig_handle, scroll, frame_idxs,
                                interval=1000 * period, repeat=False)
      #fig_handle.draw()
      fig_handle.canvas.draw_idle()
    else:
      play.anim.event_source.stop()

  # Set initial player state
  play.running = False

  # It's certainly annoying we don't seem to get auto-repeat calls of this function, 
  # e.g. when the arrow keys are held down.
  def key_press(event):
    key = event.key
    f = scroll_handle.val
    if key == 'left':
      scroll(f - 1)
    elif key == 'right':
      scroll(f + 1)
    elif key == 'pageup':
      scroll(f - big_scroll)
    elif key == 'pagedown':
      scroll(f + big_scroll)
    elif key == 'home':
      scroll(0)
    elif key == 'end':
      scroll(num_frames - 1)
    elif key == 'enter' or key == ' ':
      play(1 / play_fps)
    elif key == 'backspace':
      play(5 / play_fps)
    else:
      if key_func:
        key_func(key)

  # Register events
  scroll_handle.on_changed(draw_new)
  fig_handle.canvas.mpl_connect('key_press_event', key_press)

  # Draw initial frame
  redraw_func(0, fig_handle, axes_handle, proc_func, cmap, overlay_func=overlay_func, vmin=vmin, vmax=vmax)

  # Start playing
  play(1 / play_fps)

  # plt.show() has to be put in the end of the function,
  # otherwise, the program simply won't work, weird...
  fig_handle.show()
  #return fig_handle

#%%
def check_int_scalar(a, name):
  assert isinstance(a, int), '{} must be a int scalar, instead of {}'.format(name, type(name))


def check_callback(a, name):
  # Check http://stackoverflow.com/questions/624926/how-to-detect-whether-a-python-variable-is-a-function
  # for more details about python function type detection.
  assert callable(a), '{} must be callable, instead of {}'.format(name, type(name))

#%%
# regions - a list of region object that have a .bbox attribute, such as the skimage.RegionProperties object.
def draw_regions(axes, regions):
    for reg in regions:
        top = reg.bbox[0]
        left = reg.bbox[1]
        h = reg.bbox[2] - top
        w = reg.bbox[3] - left
        rect = mpl.patches.Rectangle((left, top), w, h, edgecolor='red', facecolor='none', linewidth=4 )
        axes.add_patch(rect)
        
#%%
# Default redraw function, specially designed for image sequences
def redraw_fn(f, fig, axes, proc_func, cmap = None, overlay_func = None, vmin=None, vmax=None):
    # Assumes proc_func returns either an image or a 2-tuple holding (image, regions)
    proc_result = proc_func(f)
    if type(proc_result) is tuple:
        (img, regions) = proc_result
    else:
        img = proc_result
        regions = None
    
    strDisp = "Frm {0}".format(f)
    # Create a generic object that's going to hold various attributes, and attach it to fig.
    # As per https://stackoverflow.com/questions/2827623/python-create-object-and-add-attributes-to-it
    if not hasattr(fig, 'redraw_fn'):
        fig.redraw_fn = lambda: None
        fig.redraw_fn.initialized = False
    # Clear any graphics items we may have drawn on the previous frame
    axes.patches.clear()
    
    if not fig.redraw_fn.initialized:
        fig.redraw_fn.im = axes.imshow(img, animated=True, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.redraw_fn.txtstyle = dict(size=20, color='cyan')
        fig.redraw_fn.txt = axes.text(0, 0, strDisp, va='top', **fig.redraw_fn.txtstyle)
        fig.redraw_fn.initialized = True
    else:
        fig.redraw_fn.im.set_array(img)
        fig.redraw_fn.txt.set_text(strDisp)
    
    if regions:
        draw_regions(axes, regions)
    
    if overlay_func:
        overlay_func(axes, f)
    return

#%%
# Example of how to use
if (__name__ == '__main__') and False:
  import numpy as np

  def redraw_fn(f, fig, axes, proc_func, cmap):
    amp = float(f) / 3000
    f0 = 3
    t = np.arange(0.0, 1.0, 0.001)
    s = amp * np.sin(2 * np.pi * f0 * t)

    if not hasattr(fig, 'redraw_fn'):
        fig.redraw_fn = lambda: None
        fig.redraw_fn.initialized = False
    if not fig.redraw_fn.initialized:
      fig.redraw_fn.l, = axes.plot(t, s, lw=2, color='red')
      fig.redraw_fn.initialized = True
    else:
      fig.redraw_fn.l.set_ydata(s)

  # The calls below open 2 animated matplotlib windows.
  videofig(100, redraw_fn)
  videofig(500, redraw_fn)
  