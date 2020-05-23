import cv2
import numpy as np
import sys
import time
from kernels import KERNELS as k


def convolve2d(image, kernel, mult):
    dim = kernel.shape[0]
    image_h = image.shape[0]
    image_w = image.shape[1]
    new_h = (image_h - dim) + 1
    new_w = (image_w - dim) + 1
    output = np.zeros((new_h, new_w))

    for x in range(dim, image_h - dim):
        for y in range(dim, image_w - dim):
            for m in range(dim):
                for n in range(dim):
                    output[x][y] += kernel[m][n] * image[x - dim + m][y - dim + n]

            output[x][y] //= mult

    return output


if __name__ == '__main__':
    start = time.time()
    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)
    new_img = convolve2d(img, kernel=k["bottom_sobel"][0], mult=k["bottom_sobel"][1])
    cv2.imwrite('./result.jpg', new_img)
    print(round(time.time() - start, 2))