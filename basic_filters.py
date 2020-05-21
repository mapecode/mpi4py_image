from mpi4py import MPI
import numpy as np
from PIL import Image
import sys


def divide_pixels(pix):
    pix_per_node = len(pix) // size
    for i in range(len(pix)):
        pix[i] = list(pix[i])

    chunks = [[] for _ in range(size)]

    for i in range(len(chunks)):
        start = pix_per_node * i
        end = start + pix_per_node
        for j in range(start, end):
            chunks[i].append(pix[j])

    return chunks


def apply_filter(filter, pix):
    R = 0
    G = 1
    B = 2

    for i in range(len(pix)):
        if filter.lower() == 'black':
            pix[i][R] = int(pix[i][R] * 0.2986 + pix[i][G] * 0.587 + pix[i][B] * 0.114) if pix[i][R] <= 255 else 255
            pix[i][G] = int(pix[i][R] * 0.2986 + pix[i][G] * 0.587 + pix[i][B] * 0.114) if pix[i][G] <= 255 else 255
            pix[i][B] = int(pix[i][R] * 0.2986 + pix[i][G] * 0.587 + pix[i][B] * 0.114) if pix[i][B] <= 255 else 255
        elif filter.lower() == 'sepia':
            pix[i][R] = int(pix[i][R] * 0.393 + pix[i][G] * 0.769 + pix[i][B] * 0.189) if pix[i][R] <= 255 else 255
            pix[i][G] = int(pix[i][R] * 0.349 + pix[i][G] * 0.686 + pix[i][B] * 0.168) if pix[i][G] <= 255 else 255
            pix[i][B] = int(pix[i][R] * 0.272 + pix[i][G] * 0.534 + pix[i][B] * 0.131) if pix[i][B] <= 255 else 255
        elif filter.lower() == 'red':
            pix[i][G] = 0
            pix[i][B] = 0
        elif filter.lower() == 'green':
            pix[i][R] = 0
            pix[i][B] = 0
        elif filter.lower() == 'blue':
            pix[i][R] = 0
            pix[i][G] = 0


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        img, filter = Image.open(sys.argv[1]), sys.argv[2]
        width, height = img.size
        pixels = divide_pixels(list(img.getdata()))
    else:
        pixels = None
        filter = None

    pixels = comm.scatter(pixels, root=0)
    filter = comm.bcast(filter, root=0)

    apply_filter(filter, pixels)

    new_pixels = comm.gather(pixels, root=0)

    if rank == 0:
        pixels = []
        for node_pixels in new_pixels:
            for pixel in node_pixels:
                pixels.append(tuple(pixel))

        image = Image.new("RGB", (width, height))
        image.putdata(pixels)
        image.save("result.jpg")
