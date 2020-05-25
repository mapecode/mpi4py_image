import numpy as np

# Sources:
# https://setosa.io/ev/image-kernels/
# https://en.wikipedia.org/wiki/Kernel_(image_processing)
# https://docs.gimp.org/2.6/es/plug-in-convmatrix.html

KERNELS = {
    "blur": [np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]]), 1],

    "box_blur": [np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]), 9],

    "gaussian_blur3x3": [np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]]), 16],

    "gaussian_blur5x5": [np.array([[1, 4, 6, 4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, 36, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]]), 256],

    "bottom_sobel": [np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]]), 1],

    "left_sobel": [np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]), 1],

    "right_sobel": [np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]), 1],

    "top_sobel": [np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]), 1],

    "emboss": [np.array([[-2, -1, 0],
                         [-1, 1, 1],
                         [0, 1, 2]]), 1],

    "identity": [np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]), 1],

    "outline": [np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]]), 1],

    "sharpen": [np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]]), 1],

    "enhance": [np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]]), 1]

}
