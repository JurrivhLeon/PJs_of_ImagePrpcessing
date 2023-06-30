import os.path as osp
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

def glb_otsu(img, nbins=256):
    hist = np.array([0]*nbins)
    for k in range(nbins):
        hist[k] = (img == k).sum()
    cdf = np.cumsum(hist)
    npixels = cdf[-1]
    cdf = cdf / npixels  # Compute the cdf from 0 to L-1, and normalize it to interval [0,1].
    acm = np.cumsum(np.array(range(nbins)) * hist / npixels)  # Compute the accumulated average from 0 to L-1.
    mean_glb = acm[-1]  # The global mean.
    w1w2 = cdf * (1 - cdf)
    w1w2[w1w2 <= 1e-6] = 1  # Preprocess data to avoid the expression below being divided by 0.
    dsq_bet = (mean_glb * cdf - acm) ** 2 / w1w2  # Compute the between-classes variance at threshold from 0 to L-1.
    thr = np.argwhere(dsq_bet == np.max(dsq_bet)).mean()  # Determine the threshold.
    img_bi = (img >= thr) * (nbins - 1)
    return img_bi

class Kmeans:
    def __init__(self, img, nbins=256):
        self.img = img
        self.nbins = nbins
        self.hist = self.compute_hist()
        self.k = None
        self.iters = None
        self.centroid = None

    def compute_hist(self): # Compute the histogram of the image.
        hist = []
        for i in range(self.nbins):
            hist.append((self.img == i).astype(int).sum())
        return np.array(hist)

    def fit(self, k: int): # Fitting.
        if k > 1:
            self.k = k
            self.centroid, self.iters = self.clustering(k)
        else:
            raise ValueError

    def clustering(self, k: int):
        # Initialize the centroids.
        centroid = [np.random.choice(self.img.flatten()).astype(int)]
        pix = list(set(self.img.flatten()))
        for j in range(k - 1):
            min_dist = []
            for i in pix:
                min_dist.append(np.abs(i - np.array(centroid)).min())
            centroid.append(pix[np.array(min_dist).argmax()])
        centroid = np.array(centroid).astype(float)

        # Clustering.
        done = 0
        iters = 0
        while not done:
            iters += 1
            # Assign each pixel to its nearest cluster, represented by centroid.
            label = np.zeros((self.nbins,), dtype=int)  # The cluster label of a pixel with given value.
            cluster = [[] for j in range(self.k)] # Initialize clusters.
            for i in range(self.nbins):
                label[i] = np.abs(i - centroid).argmin()
                cluster[label[i]].append(i)
            done = 1

            # Update centroids.
            for j in range(self.k):
                centroid_j = (self.hist[cluster[j]] * np.array(cluster[j])).sum()
                centroid_j = float(centroid_j) / float(self.hist[cluster[j]].sum())
                if np.abs(centroid_j - centroid[j]) > 1e-6:
                    done = 0
                    centroid[j] = centroid_j

        return np.sort(centroid), iters

    def segment(self):
        # Do segmentation, and return a matrix with each entry the label of the pixel.
        segmentation = np.zeros_like(self.img)
        for i in range(self.nbins):
            segmentation[self.img == i] = np.abs(i - self.centroid).argmin()
        return segmentation

# Visualize the segmentation result.
def vis_seg(seg, nbins=256):
    seg = (seg - seg.min()) / (seg.max() - seg.min()) * (nbins - 1)
    return cv2.applyColorMap(seg.astype(np.uint8), cv2.COLORMAP_PINK)


def main():
    parser = argparse.ArgumentParser(description='Segmentation.')
    parser.add_argument('-i', '--img_path', type=str, default='images/brain.jpg',
                        help='the path of source image.')
    parser.add_argument('-k', '--ncluster', type=int, default=2,
                        help='Number of clusters.')
    args = parser.parse_args()

    img = cv2.imread(args.img_path, 0)
    seg_model = Kmeans(img)
    seg_model.fit(args.ncluster)
    segmentation = seg_model.segment()
    vis = vis_seg(segmentation)

    img_thr = glb_otsu(img)

    # Visualization.
    plt.figure()

    plt.subplot(121)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.title('Segmentation')
    plt.imshow(vis[:, :, ::-1])
    plt.axis('off')

    plt.subplots_adjust(left=0.10, right=0.90, wspace=0.35, )
    plt.show()

    cv2.imwrite(osp.splitext(args.img_path)[0] + f'_seg_{args.ncluster}.jpg', vis)
    cv2.imwrite(osp.splitext(args.img_path)[0] + f'_otsu.jpg', img_thr)



if __name__ == '__main__':
    main()
