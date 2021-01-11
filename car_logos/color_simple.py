"""
Proste deskryptory koloru
"""

from PIL import Image
import cv2
import numpy as np
import scipy.ndimage as ndimage


def dominant_color_desc(filename):
    img = Image.open(filename)
    img = img.copy()
    img.thumbnail((150, 150))

    # Reduce to palette
    paletted = img.convert("P", palette=Image.ADAPTIVE, colors=10)

    # Find dominant colors
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    colors = list()
    for i in range(10):
        palette_index = color_counts[i][1]
        colors.append(palette[palette_index * 3 : palette_index * 3 + 3][0])
    return colors


def color_mean_desc(filename):
    img = cv2.imread(filename)
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


def color_variance_desc(filename):
    img = cv2.imread(filename)
    lbl, nlbl = ndimage.label(img)
    out = ndimage.variance(img, lbl, index=np.arange(1, nlbl + 1))
    return out


def color_hist(filename):
    """
    Tworzy wektor gdzie kolejno są histogramy 32 elementowe niebieskiego, zielonego i czerwonego kanału
    """
    img = cv2.imread(filename)
    blue = cv2.calcHist([img], [0], mask=None, histSize=[32], ranges=[0, 256])
    green = cv2.calcHist([img], [1], mask=None, histSize=[32], ranges=[0, 256])
    red = cv2.calcHist([img], [2], mask=None, histSize=[32], ranges=[0, 256])

    hist = np.ravel(blue).tolist() + np.ravel(green).tolist() + np.ravel(red).tolist()
    return hist

def color_layout_desc(filename):
    pass

def main():
    file = "./dataset/tests/1.jpg"
    color_hist(file)
    # print(dominant_color_desc(file))
    # print(color_mean_desc(file))
    # print(color_variance_desc(file))


if __name__ == "__main__":
    main()
