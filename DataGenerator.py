import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def _get_bounds_(image):

    '''
    helper function that finds the
    top and bottom indices for the row values that
    encompasses an image
    '''
    y = [] #list of row indices that have at least one non zero value
    for i, row in enumerate(image.reshape(28,28)):
        if np.any(row > 70/):
            y.append(i)
    return y[0], y[-1]

def cut_image(n, X, Y):
    '''
    removes empty area from image (as much as possible)
    '''
    image = X[n].reshape(28,28)
    y1, y2 = _get_bounds_(image)
    x1, x2 = _get_bounds_(image.T)

    return image[y1:y2+1,x1:x2+1]

def generate_image(X, Y):
    '''
    generates a random sequence of handwritten numbers
    to a 30 * 200 canvas
    @param numpy.array X: array containing images
    @param numpy.array T: array containing corresponding labels
    @rtype tuple: (canvas, x_vals, y_vals)
        canvas: Generated image
        x_vals: list of tuples representing value of bounding boxes
        y_vals: list of labels for each digit in order
    '''
    H = 30 #Height of the canvas
    W = 200 #Width of the canvas

    N = np.random.randint(1,6) #the number of digits to be written

    sep_len = W // N # the width of space each digit will be in

    M = len(Y) # Number of data points to sample from
    y_vals = []
    x_vals = [] #list of tuples describing each digit image (x position, y position, width, height)
    canvas = np.zeros((40,200))

    for i in range(N):

        sample_index = np.random.randint(M)
        y_vals.append(Y[sample_index])
        digit = cut_image(sample_index, X, Y)

        #where to place the image
        digit_h, digit_w = digit.shape
        top_right_point_x = np.random.randint(sep_len//2) + i*sep_len
        top_right_point_y = np.random.randint(H//2)

        x_vals.append((top_right_point_x, top_right_point_y, digit_h, digit_w))



        canvas[top_right_point_y:top_right_point_y+digit_h,
               top_right_point_x:top_right_point_x+digit_w] += digit

    return canvas, x_vals, y_vals

def generate_data(N, X, Y):
    '''

    '''
    pass
