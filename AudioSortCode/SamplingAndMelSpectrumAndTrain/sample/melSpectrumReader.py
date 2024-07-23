import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Entry, Label, filedialog

class MelSpectrogramViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Mel Spectrogram Viewer")
        
        self.file_list = []
        self.current_index = 0

        self.load_button = Button(master, text="Load Directory", command=self.load_directory)
        self.load_button.pack()

        self.prev_button = Button(master, text="Previous", command=self.show_previous)
        self.prev_button.pack()

        self.next_button = Button(master, text="Next", command=self.show_next)
        self.next_button.pack()

        self.jump_label = Label(master, text="Jump to page:")
        self.jump_label.pack()

        self.page_entry = Entry(master)
        self.page_entry.pack()

        self.jump_button = Button(master, text="Go", command=self.jump_to_page)
        self.jump_button.pack()

        self.figure, self.ax = plt.subplots()
        self.canvas = None

    def load_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
            self.file_list.sort()
            self.current_index = 0
            if self.file_list:
                self.show_image()

    def show_image(self):
        if self.file_list:
            self.ax.clear()
            data = np.load(self.file_list[self.current_index])
            self.ax.imshow(data, aspect='auto', origin='lower')
            self.ax.set_title(f"File: {os.path.basename(self.file_list[self.current_index])}")
            self.figure.canvas.draw()

    def show_previous(self):
        if self.file_list and self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next(self):
        if self.file_list and self.current_index < len(self.file_list) - 1:
            self.current_index += 1
            self.show_image()

    def jump_to_page(self):
        try:
            page = int(self.page_entry.get()) - 1
            if 0 <= page < len(self.file_list):
                self.current_index = page
                self.show_image()
        except ValueError:
            pass

if __name__ == "__main__":
    root = Tk()
    viewer = MelSpectrogramViewer(root)
    plt.ion()
    plt.show()
    root.mainloop()
