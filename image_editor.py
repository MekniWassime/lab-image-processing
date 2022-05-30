import tkinter as tk
from PIL import Image, ImageTk
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter import *
from utility import *


image = readPGM("./assets/portraitWritten.pgm")
image_stack = []

normal_state = tk.ACTIVE
binary_state = tk.DISABLED

def reset_states():
    global normal_state
    global binary_state
    normal_state = tk.DISABLED
    binary_state = tk.ACTIVE
    switch_states()

def switch_states():
    global normal_state
    global binary_state
    temp = normal_state
    normal_state = binary_state
    binary_state = temp
    for child in side_frame.winfo_children():
        if child['state'] == normal_state:
            child['state'] = binary_state
        else:
            child['state'] = normal_state

def update_image(newimage, append_to_stack = True):
    global image
    global image_stack
    if(append_to_stack):
        image_stack.append(image)
    image = newimage
    im.set_data(image)
    canvas.draw()
    hist = histogram(image)
    ax1.clear()
    ax1.plot(hist)
    bar1.draw()

def undo():
    global image_stack
    newimage = image_stack.pop()
    update_image(newimage, False)

root= tk.Tk()
button1 = tk.Button(root, text='Open', command=lambda:[update_image(open_file()), switch_states()])
button2 = tk.Button(root, text='Save As', command=lambda:save_file(image))#implement this
button3 = tk.Button(root, text='Undo', command=undo)
button1.grid(row=0, column=1)
button2.grid(row=0, column=2)
button3.grid(row=0, column=3)

fig = plt.figure(1,figsize=(6,4))
im = plt.imshow(image, cmap="gray") # later use a.set_data(new_data)
ax = plt.gca()
ax.set_xticklabels([]) 
ax.set_yticklabels([]) 
plt.close(1)
# a tk.DrawingArea
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=1, column=1,columnspan=3)

hist = histogram(image)
figure1 = plt.Figure(figsize=(6,3), dpi=100)
ax1 = figure1.add_subplot(111)
bar1 = FigureCanvasTkAgg(figure1, root)
bar1.get_tk_widget().grid(row=2, column=1, columnspan=3)
ax1.plot(hist)

side_frame = Frame(root)
tk.Button(side_frame, text="Cummulative Histogram", command=lambda:histogram_cummulative(image), state=normal_state).pack()
tk.Button(side_frame, text="Equalize Histogram", command=lambda:update_image(equalize_histogram(image)), state=normal_state).pack()
tk.Button(side_frame, text="Linear Transformation", command=lambda:update_image(linear_transformation(root, image)), state=normal_state).pack()
tk.Button(side_frame, text="Salt And Pepper Noise", command=lambda:update_image(addNoise(image)), state=normal_state).pack()
tk.Button(side_frame, text="Average Filter", command=lambda:update_image(filter_average(image)), state=normal_state).pack()
tk.Button(side_frame, text="Median Filter", command=lambda:update_image(filter_median(image)), state=normal_state).pack()
tk.Button(side_frame, text="Sharpen Filter", command=lambda:update_image(filter_mask(image)), state=normal_state).pack()
tk.Button(side_frame, text="Manual Segmentation", command=lambda:[update_image(manual_segmentation(image)), switch_states()], state=normal_state).pack()
tk.Button(side_frame, text="Automatic Segmentation", command=lambda:[update_image(auto_segmentation(image)), switch_states()], state=normal_state).pack()
tk.Button(side_frame, text="Erosion", command=lambda:update_image(manual_erosion(image)), state=binary_state).pack()
tk.Button(side_frame, text="Dilation", command=lambda:update_image(manual_dilation(image)), state=binary_state).pack()
tk.Button(side_frame, text="Open", command=lambda:update_image(manual_open(image)), state=binary_state).pack()
tk.Button(side_frame, text="Close", command=lambda:update_image(manual_close(image)), state=binary_state).pack()

side_frame.grid(rowspan=3, row=0, column=0)

root.mainloop()