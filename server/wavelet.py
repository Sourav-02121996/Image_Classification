"""Author: Sourav Das

Wavelet transform helper to extract high-frequency image features.
"""

import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # Convert to grayscale for wavelet processing.
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    # Convert to float and normalize to [0, 1].
    imArray =  np.float32(imArray)
    imArray /= 255;
    # Compute wavelet coefficients.
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    # Process coefficients: remove approximation to keep detail only.
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # Reconstruct image from detail coefficients.
    imArray_H=pywt.waverec2(coeffs_H, mode);
    # Scale back to 8-bit image range.
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H
