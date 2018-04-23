# if you have iterm2 for osx (www.iterm2.com) this is a like print(...) for images in the console

import base64
import io
import numpngw
import numpy as np

def show_image(a, offset=None, scale=None):
    # if not uint8 we assume mean = 0 and ranges from [-1..1]
    #if a.dtype != np.uint8:
    ##    a += offset
    #    a *= scale
    #    a = a.astype(np.uint8)

    if a.max() <= 1. or a.min() < 0.: # assume not [0-255] 
        suggested_offset, suggested_scale = 0., 255 # assumes input [0,1]
    else:
        suggested_offset, suggested_scale = 0, 1.   # assumes input [0.255]

    if offset is None or scale is None:
        offset, scale = suggested_offset, suggested_scale

    png_array = io.BytesIO()
    numpngw.write_png(png_array, ((a + offset) * scale).astype(np.uint8) if a.dtype != np.uint8 else a)
    encoded_png_array =base64.b64encode(png_array.getvalue()).decode("utf-8", "strict")  
    png_array.close()
    image_seq = '\033]1337;File=[width=auto;height=auto;inline=1]:'+encoded_png_array+'\007'
    print(image_seq, end='')
