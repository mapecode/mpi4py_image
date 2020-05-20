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
        start = pix_per_node*i
        end = start + pix_per_node
        for j in range(start, end):
            chunks[i].append(pix[j])

    return chunks


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(size)
        img = Image.open(sys.argv[1])
        width, height = img.size
        pixels = divide_pixels(list(img.getdata()))
    else:
        pixels = None

    pixels = comm.scatter(pixels, root=0)

    print(rank, len(pixels))

    for i in range(len(pixels)):
        pixels[i][0] = 0

    new_pixels = comm.gather(pixels, root=0)

    if rank == 0:
        pixels = []
        for node_pixels in new_pixels:
            for pixel in node_pixels:
                pixels.append(tuple(pixel))

        image = Image.new("RGB", (width, height))
        image.putdata(pixels)
        image.save("test.jpg")



