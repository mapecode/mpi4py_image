import cv2
import numpy as np
import sys
import time
from mpi4py import MPI
from PIL import Image
from kernels import KERNELS as k


def create_new_pixels(new_h, new_w):
    full_output = np.zeros((new_h, new_w), dtype=np.int32)
    pix_per_node = new_h // size
    rest_pix = new_h % size

    chunks = [[] for _ in range(size)]

    for i in range(len(chunks)):
        start = pix_per_node * i
        end = start + pix_per_node if i != len(chunks) - 1 else start + pix_per_node + rest_pix
        chunks[i] = full_output[start:end]

    return chunks


def join_pixels(new_pixels):
    pixels = np.zeros((new_h, new_w), dtype=np.int32)

    i = 0

    for node_pixels in new_pixels:
        pixels[i: i + len(node_pixels)] = node_pixels
        i += len(node_pixels)

    return pixels


def convolve2d(image, output, kernel, mult):
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
    pix_per_node = new_h // size
    rest_pix = new_h % size
    start = pix_per_node * rank
    end = start + pix_per_node if rank != size - 1 else start + pix_per_node + rest_pix

    for x in range(start, end):
        for y in range(dim, image_w - dim):
            output[x-start][y] = mult_matrix(kernel, image, x, y, 0, dim)
            output[x-start][y] //= mult

    return output


if __name__ == '__main__':
    # start = time.time()
    # img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)
    # new_img = convolve2d(img, kernel=k["emboss"][0], mult=k["emboss"][1])
    # cv2.imwrite('./result.jpg', new_img)
    # print(round(time.time() - start, 2))
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start_time = time.time()
        img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)
        kernel = k[sys.argv[2]]
        dim = kernel[0].shape[0]
        image_h = img.shape[0]
        image_w = img.shape[1]
        new_h = (image_h - dim) + 1
        new_w = (image_w - dim) + 1
        empty_pixels = create_new_pixels(new_h, new_w)
    else:
        empty_pixels = None
        img = None
        kernel = None

    empty_pixels = comm.scatter(empty_pixels, root=0)
    img = comm.bcast(img, root=0)
    kernel = comm.bcast(kernel, root=0)

    new_pixels_div = convolve2d(img, empty_pixels, kernel=kernel[0], mult=kernel[1])

    new_pixels = comm.gather(new_pixels_div, root=0)

    if rank == 0:
        new_img = join_pixels(new_pixels)
        cv2.imwrite('./result.jpg', new_img)
        print(round(time.time() - start_time, 2))
