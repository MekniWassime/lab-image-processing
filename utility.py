import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog as fd
from tkinter.simpledialog import *
import tkinter as tk;

def open_file():
    filename = fd.askopenfilename()
    if(filename):
        image = readPGM(filename)
        return image

def writePGM(matrix, name):
    with open(name, "wb") as f:
        f.write(b'P5\n')
        height, width = matrix.shape
        max_grayscale_value = 255
        f.write(str(width).encode("utf-8") + b' ' + str(height).encode("utf-8") + b'\n' + str(max_grayscale_value).encode("utf-8") + b'\n')
        for row in range(0, height):
            for col in range(0, width):
                f.write(int(matrix[row, col]).to_bytes(1, "big"))

def save_file(image):
    filename = fd.asksaveasfilename()
    writePGM(image, filename)

def test(canvas, im):
    im.set_data(readPGM("./assets/portraitCigare.pgm"))
    canvas.draw()

def readPGM(image_path):
    with open(image_path, "rb") as f:
        assert(f.read(1) == b'P')
        assert(f.read(1) in [b'2',b'5'])
        assert(f.read(1) == b'\n')
        #ignore comments
        byte = f.read(1)
        if(byte == b'#'):
            while(not byte == b'\n'):
                byte = f.read(1)
        #read dimentions
        width = b''
        while(not byte == b' '):
            width += byte
            byte = f.read(1)
        width = int(width.decode("utf-8"))
        height = b''
        byte = f.read(1)
        while(not byte == b'\n'):
            height += byte
            byte = f.read(1)
        height = int(height.decode("utf-8"))
        max_grayscale_value = b''
        byte = f.read(1)
        while(not byte == b'\n'):
            max_grayscale_value += byte
            byte = f.read(1)
        max_grayscale_value = int(max_grayscale_value.decode("utf-8"))
        array = np.empty((height, width), dtype=float)
        for row in range(0, height):
            for col in range(0, width):
                array[row, col] = int.from_bytes(f.read(1), "big")
        return array

def histogram(image):
    hist,bin = np.histogram(image.ravel(),256,[0,255])
    return hist

def histogram_cummulative(image, show=True):
    hist = histogram(image)
    hc = np.zeros(256,int)
    hc[0] = hist[0]
    for index in range(1,len(hist)): 
        hc[index]= hist[index] + hc[index -1]
    if(show):
        plt.figure(figsize=(6,4))
        plt.xlim([0,255])
        plt.plot(hc)
        plt.title('histogram cumulative')
        plt.show()
    return hc

def normalize_histogram(image):
    height, width = image.shape
    nb_pixels = height * width
    hn = histogram_cummulative(image, show=False) / nb_pixels * 255
    return hn

def equalize_histogram(image):
    histogram_normalized = normalize_histogram(image)
    eq_image = np.empty(image.shape, dtype=float)
    height, width = image.shape
    for row in range(0,height):
        for col in range(0,width):
            eq_image[row,col] = int(histogram_normalized[int(image[row,col])])
    return eq_image


def plot_linear_transformation(p1,p2):
    plt.plot([0, p1[0], p2[0], 255], [0, p1[1], p2[1], 255], 'r', linewidth=4)
    plt.plot([0,255], [0, 255], 'k:')
    plt.show()


def linear_transformation(root, image):
   min = int(image.min())
   max = int(image.max())
   top= Toplevel(root)
   top.title("Linear Transformation")
   tk.Label(top, text= "Enter the coordinates of points").grid(columnspan=3)
   tk.Label(top, text= "Point 1: ").grid(row=1,column=0)
   var1 = tk.StringVar()
   entry1 = tk.Entry(top, textvariable=var1)
   entry1.insert(0, str(min))
   entry1.grid(row=1, column=1)
   var2 = tk.StringVar()
   entry2 = tk.Entry(top, textvariable=var2)
   entry2.insert(0, "0")
   entry2.grid(row=1, column=2)
   tk.Label(top, text= "Point 2: ").grid(row=2,column=0)
   var3 = tk.StringVar()
   entry3 = tk.Entry(top, textvariable=var3)
   entry3.insert(0, str(max))
   entry3.grid(row=2, column=1)
   var4 = tk.StringVar()
   entry4 = tk.Entry(top, textvariable=var4)
   entry4.insert(0, "255")
   entry4.grid(row=2, column=2)
   top.wait_window()
   result = ([int(var1.get()), int(var2.get())],[int(var3.get()),int(var4.get())])
   p1, p2 = result
   plot_linear_transformation(p1, p2)
   return apply_linear_transformation(image, p1, p2)

def apply_linear_transformation(image,p1,p2):
    image_enhanced = np.empty(image.shape, dtype=float)
    height, width = image.shape
    b2 = (p2[0] * p1[1] - p1[0] * p2[1]) / (p2[0] - p1[0])
    b3 = (255 * p2[1] - 255 * p2[0]) / (255 - p2[0])
    for row in range(0,height):
        for col in range(0,width):
            if (image[row,col] <p1[0]):
                image_enhanced[row,col] = (p1[1]/p1[0]) * image[row,col]
            elif (image[row,col] <p2[0]):
                image_enhanced[row,col] = ( ((p2[1] - p1[1])/ (p2[0] - p1[0])) * image[row,col] )+b2
            else:
                image_enhanced[row,col] = ( ((255 - p2[1])/ (255- p2[0])) * image[row,col] ) + b3
    return image_enhanced

import random
def addNoise(matrix):
    height, width = matrix.shape
    result = np.empty(matrix.shape, dtype=float)
    for row in range(0, height):
            for col in range(0, width):
                random_int = random.randint(0, 20)
                if(random_int == 0):
                    result[row, col] = 0
                elif(random_int == 20):
                    result[row, col] = 255
                else:
                    result[row, col] = matrix[row, col]
    return result

def get_pixel_with_mirror(matrix, row, col):
    height, width = matrix.shape
    height -= 1
    width -= 1
    if(row<0):
        row = -row
    if(row>height):
        row = height - (row - height)
    if(col<0):
        col = -col
    if(col>width):
        col = width - (col - width)
    return matrix[row, col]

def filter_average(matrix):
    n = tk.simpledialog.askfloat("Average filter", "Please enter the size of the filter")
    n = int(n)
    height, width = matrix.shape
    result = np.empty(matrix.shape, dtype=float)
    for row in range(0, height):
        for col in range(0, width):
            sum = 0
            divide_by = 0
            for x in range(row - n//2, row + n//2 + 1):
                for y in range(col - n//2, col + n//2 + 1):
                    divide_by+=1
                    sum += get_pixel_with_mirror(matrix, x, y)
            moy = sum / divide_by
            result[row, col] = moy
    return result

def filter_median(matrix):
    n = tk.simpledialog.askfloat("Median filter", "Please enter the size of the filter")
    n = int(n)
    height, width = matrix.shape
    result = np.empty(matrix.shape, dtype=float)
    for row in range(0, height):
        for col in range(0, width):
            array = []
            for x in range(row - n//2, row + n//2):
                for y in range(col - n//2, col + n//2):
                    array.append(get_pixel_with_mirror(matrix, x, y))
            result[row, col] = np.median(array)
    return result

def filter_mask(matrix):
    mask = np.array([[1, -2, 1 ],
                    [-2, 5, -2],
                    [1, -2, 1 ],])
    height, width = matrix.shape
    n, m = mask.shape
    result = np.empty(matrix.shape, dtype=float)
    for row in range(0, height):
        for col in range(0, width):
            sum = 0
            for x in range(row - n//2, row + n//2 + 1):
                for y in range(col - m//2, col + m//2 + 1):
                    mask_x = x - row + n//2
                    mask_y = y - col + m//2
                    sum += get_pixel_with_mirror(matrix, x, y)*mask[mask_x,mask_y]
            result[row, col] = sum
    return result

import math
def signal_to_noise_ratio(original, filtered):
    original = original.flatten()
    filtered = filtered.flatten()
    original_avg = np.average(original)
    snr = math.sqrt((sum(np.square(original - original_avg)))/(sum(np.square(filtered - original))))
    return snr

def manual_segmentation(image):
    n = tk.simpledialog.askfloat("Segmentation", "Please enter segmentation threshold")
    n = int(n)
    return segmentation(image, n)

def segmentation(image, threshold):
    height, width = image.shape
    result = np.empty(image.shape, dtype=float)
    for row in range(0, height):
        for col in range(0, width):
            result[row, col] = 0 if(image[row,col]<threshold) else 255
    return result

def auto_segmentation(image):
    hist = histogram(image).flatten()
    min_variance = math.inf
    best_threshold = 0
    for threshold in range(1,255):
        low_hist = hist[hist < threshold]
        high_hist = hist[hist > threshold]
        low_variance = low_hist.var()
        high_variance = high_hist.var()
        avg_variance = (low_variance + high_variance) / 2
        if(avg_variance < min_variance):
            best_threshold = threshold
            min_variance = avg_variance
    print(f"chosen threshold = {best_threshold}")
    return segmentation(image, best_threshold)

def erosion(image, n):
    height, width = image.shape
    result = np.empty(image.shape, dtype=float)
    for row in range(0, height):
        for col in range(0, width):
            found_zero = False
            if(image[row,col]==0):
                for x in range(row - n//2, row + n//2 + 1):
                    for y in range(col - n//2, col + n//2 + 1):
                        if(not found_zero and x >= 0 and y >= 0 and x < height and y < width and x!=row and y!=col and image[x,y]==255):
                            found_zero = True
            else:
                found_zero = True                 
            result[row, col] = 255 if found_zero else 0
    return result

def manual_dilation(image):
    n = tk.simpledialog.askfloat("Dilation", "Please enter the square")
    n = int(n)
    return dilation(image, n)

def manual_erosion(image):
    n = tk.simpledialog.askfloat("Erosion", "Please enter the square")
    n = int(n)
    return erosion(image, n)

def manual_open(image):
    n = tk.simpledialog.askfloat("Open", "Please enter the square")
    n = int(n)
    return dilation(erosion(image, n), n)

def manual_close(image):
    n = tk.simpledialog.askfloat("Close", "Please enter the square")
    n = int(n)
    return erosion(dilation(image, n), n)

def dilation(image, n):
    height, width = image.shape
    result = np.empty(image.shape, dtype=float)
    for row in range(0, height):
        for col in range(0, width):
            found_zero = False
            if(image[row,col]==255):
                for x in range(row - n//2, row + n//2 + 1):
                    for y in range(col - n//2, col + n//2 + 1):
                        if(not found_zero and x >= 0 and y >= 0 and x < height and y < width and x!=row and y!=col and image[x,y]==0):
                            found_zero = True
            else:
                found_zero = True                 
            result[row, col] = 0 if found_zero else 255
    return result


