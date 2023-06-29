import numpy as np


class local_histogram:
    def __init__(self, img, size, kernel=[0, 0], nbins=256):
        self.img = img
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.kernel = kernel  # the coordinate of processing pixel.
        self.size = size
        self.nbins = nbins
        self.hist = np.array([0]*nbins)
        self.compute()
        self.done = 0

    def compute(self):  # Compute the initial local histogram.
        if self.kernel[0] in range(self.height) and self.kernel[1] in range(self.width):
            i_start = max(0, self.kernel[0] - self.size)
            j_start = max(0, self.kernel[1] - self.size)
            i_end = min(self.kernel[0] + self.size, self.height - 1)
            j_end = min(self.kernel[1] + self.size, self.width - 1)
        else:
            raise ValueError
        nbhd = self.img[i_start:i_end+1, j_start:j_end+1]
        for k in np.unique(nbhd):
            self.hist[k] += np.sum(nbhd == k)

    def update(self):  # Update the local histogram on each pixel step.
        if self.done:
            return 1
        i_start = max(0, self.kernel[0] - self.size)
        j_start = max(0, self.kernel[1] - self.size)
        i_end = min(self.kernel[0] + self.size, self.height - 1)
        j_end = min(self.kernel[1] + self.size, self.width - 1)
        if self.kernel[0] % 2:
            # Consider the case in odd row.
            if self.kernel[1] == 0:
                # If the kernel is already on left border, walk down to next row.
                self.kernel[0] += 1
                if self.kernel[0] == self.height:  # Test if all rows are finished.
                    self.done = 1
                else:
                    if self.kernel[0] > self.size:  # Identify the deleted row.
                        del_row = self.img[i_start, j_start:j_end+1]
                        for v in del_row:
                            self.hist[v] -= 1  # Update the histogram.
                    if self.kernel[0] < self.height - self.size:  # Identify the added row.
                        add_row = self.img[i_end+1, j_start:j_end+1]
                        for v in add_row:
                            self.hist[v] += 1  # Update the histogram.
            else:   # The kernel hasn't reached the left border, walk to left one pixel.
                self.kernel[1] -= 1
                if self.kernel[1] >= self.size:  # Identify the added column.
                    add_col = self.img[i_start:i_end+1, j_start-1]
                    for v in add_col:
                        self.hist[v] += 1  # Update the histogram.
                if self.kernel[1] < self.width - self.size - 1:  # Identify the deleted column.
                    del_col = self.img[i_start:i_end+1, j_end]
                    for v in del_col:
                        self.hist[v] -= 1  # Update the histogram.
        else:
            # The case in even row.
            if self.kernel[1] == self.width - 1:
                # If the kernel is already on right border, walk down to next row.
                self.kernel[0] += 1
                if self.kernel[0] == self.height:
                    self.done = 1
                else:
                    if self.kernel[0] > self.size:  # Identify the deleted row.
                        del_row = self.img[i_start, j_start:j_end+1]
                        for v in del_row:
                            self.hist[v] -= 1  # Update the histogram.
                    if self.kernel[0] < self.height - self.size:  # Identify the added row.
                        add_row = self.img[i_end+1, j_start:j_end+1]
                        for v in add_row:
                            self.hist[v] += 1  # Update the histogram.
            else:
                # The kernel hasn't reached the right border, walk to right one pixel.
                self.kernel[1] += 1
                if self.kernel[1] > self.size:
                    # Identify the deleted column.
                    del_col = self.img[i_start:i_end+1, j_start]
                    for v in del_col:
                        self.hist[v] -= 1  # Update the histogram.
                if self.kernel[1] < self.width - self.size:
                    # Identify the added column.
                    add_col = self.img[i_start:i_end+1, j_end+1]
                    for v in add_col:
                        self.hist[v] += 1  # Update the histogram.
