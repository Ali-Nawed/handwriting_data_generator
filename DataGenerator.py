"""
Custom data generatiion library.

Samples data from csv file of mnist digits and places them on a canvas
returns an image, a list of bounding boxes and a list of target values
representing the digit within each bounding box.
"""

import numpy as np
import cv2


def _get_bounds_(image):

    """Find the top and bottom indices for the row values that encompasses an image."""

    row_list = [] #list of row indices that have at least one non zero value
    for i, row in enumerate(image.reshape(28, 28)):
        if np.any(row > 70):
            row_list.append(i)
    return row_list[0], row_list[-1]

def cut_image(n, X, Y):
    """removes empty area from image (as much as possible)"""

    image = X[n].reshape(28, 28)
    top_y, bottom_y = _get_bounds_(image)
    left_x, right_x = _get_bounds_(image.T)

    return image[top_y:bottom_y+1, left_x:right_x+1]

def generate_image(X, Y):
    """
    Return a random sequence of handwritten numbers.

    to a 30 * 200 canvas
    @param numpy.array X: array containing images
    @param numpy.array T: array containing corresponding labels
    @rtype tuple: (canvas, x_vals, y_vals)
        canvas: Generated image
        x_vals: list of tuples representing value of bounding boxes
        y_vals: list of labels for each digit in order
    """

    height = 30
    width = 200

    num_digits = np.random.randint(1,6)

    sep_len = width // num_digits # the width of space each digit will be in

    M = len(Y)
    y_vals = []
    x_vals = [] #list of tuples describing each digit image (x position, y position, width, height)
    canvas = np.zeros((40,200))

    for i in range(num_digits):

        sample_index = np.random.randint(M)
        y_vals.append(Y[sample_index])
        digit = cut_image(sample_index, X, Y)

        #where to place the image
        digit_h, digit_w = digit.shape
        top_right_point_x = np.random.randint(sep_len//2) + i*sep_len
        top_right_point_y = np.random.randint(height//2)

        x_vals.append((top_right_point_x, top_right_point_y, digit_h, digit_w))

        canvas[top_right_point_y:top_right_point_y+digit_h,
               top_right_point_x:top_right_point_x+digit_w] += digit

    return canvas, x_vals, y_vals

def show_boxes(image, box_list):
    """Plot image with bounding boxes."""
    
    pass

def generate_data(N, X, Y):
    """Generate custom dataset."""

    pass
