import cv2
import numpy as np
import sys
import time
from kernels import KERNELS as k


def convolve2d(image, kernel, mult):
    def mult_matrix(kernel, image, x, y, i, dim):
        if i < dim * dim:
            m = i // dim
            n = i % dim
            result = kernel[m][n] * image[x - dim + m][y - dim + n]
            return result + mult_matrix(kernel, image, x, y, i + 1, dim)
        else:
            return 0

    dim = kernel.shape[0]
    image_h = image.shape[0]
    image_w = image.shape[1]
    new_h = (image_h - dim) + 1
    new_w = (image_w - dim) + 1
    output = np.zeros((new_h, new_w))

    for x in range(dim, image_h - dim):
        for y in range(dim, image_w - dim):
            output[x][y] = mult_matrix(kernel, image, x, y, 0, dim)
            output[x][y] //= mult

    return output


if __name__ == '__main__':
    start = time.time()
    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)
    new_img = convolve2d(img, kernel=k["emboss"][0], mult=k["emboss"][1])
    cv2.imwrite('./result.jpg', new_img)
    print(round(time.time() - start, 2))
