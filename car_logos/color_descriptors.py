"""
Proste deskryptory koloru
"""

from PIL import Image
import cv2
import numpy as np
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans


def dominant_color_desc(filename):
    # https://gist.github.com/skt7/71044f42f9323daec3aa035cd050884e
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(4)
    kmeans.fit(img)
    res = kmeans.cluster_centers_
    res = res.ravel().astype(int)
    return res


def dominant_color_2_desc(filename):
    # https://gist.github.com/zollinger/1722663
    # https://stackoverflow.com/a/61730849/3978701

    img = Image.open(filename)
    img = img.copy()
    img.thumbnail((150, 150))

    # Reduce to palette
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=10)

    # Find dominant colors
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    res = []
    for i in range(4):
        palette_index = color_counts[i][1]
        res.append(palette[palette_index * 3:palette_index * 3 + 3][0])

    return res


def color_mean_desc(filename):
    img = cv2.imread(filename)
    mean_row = np.average(img, axis=0)
    mean = np.average(mean_row, axis=0)
    return mean


def color_hist_rgb_desc(filename):
    """
    Tworzy wektor gdzie kolejno są histogramy 32 elementowe niebieskiego, zielonego i czerwonego kanału
    """
    img = cv2.imread(filename)
    blue = cv2.calcHist([img], [0], mask=None, histSize=[32], ranges=[0, 256])
    green = cv2.calcHist([img], [1], mask=None, histSize=[32], ranges=[0, 256])
    red = cv2.calcHist([img], [2], mask=None, histSize=[32], ranges=[0, 256])

    hist = np.ravel(blue).tolist() + np.ravel(green).tolist() + np.ravel(red).tolist()
    return hist


def color_hist_hsv_desc(filename):
    """
    Histogram w przestrzeni HSV
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = cv2.calcHist([img], [0], mask=None, histSize=[32], ranges=[0, 256])
    saturation = cv2.calcHist([img], [1], mask=None, histSize=[32], ranges=[0, 256])
    value = cv2.calcHist([img], [2], mask=None, histSize=[32], ranges=[0, 256])

    hist = np.ravel(hue).tolist() + np.ravel(saturation).tolist() + np.ravel(value).tolist()
    return hist


def color_layout_desc(filename, rows=4, cols=4):
    # https://github.com/tarunlnmiit/irtex-1.0/blob/master/color_layout_descriptor/CLDescriptor.py
    img = cv2.imread(filename)
    img = cv2.resize(img, (128, 128))
    averages = np.zeros((rows, cols, 3))
    imgH, imgW, _ = img.shape
    for row in range(rows):
        for col in range(cols):
            slice = img[imgH // rows * row: imgH // rows * (row + 1),
                    imgW // cols * col: imgW // cols * (col + 1)]
            average_color_per_row = np.mean(slice, axis=0)
            average_color = np.mean(average_color_per_row, axis=0)
            average_color = np.uint8(average_color)
            averages[row][col][0] = average_color[0]
            averages[row][col][1] = average_color[1]
            averages[row][col][2] = average_color[2]
    icon = cv2.cvtColor(
        np.array(averages, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(icon)
    dct_y = cv2.dct(np.float64(y))
    dct_cb = cv2.dct(np.float64(cb))
    dct_cr = cv2.dct(np.float64(cr))
    dct_y_zigzag = []
    dct_cb_zigzag = []
    dct_cr_zigzag = []
    flip = True
    flipped_dct_y = np.fliplr(dct_y)
    flipped_dct_cb = np.fliplr(dct_cb)
    flipped_dct_cr = np.fliplr(dct_cr)
    for i in range(rows + cols - 1):
        k_diag = rows - 1 - i
        diag_y = np.diag(flipped_dct_y, k=k_diag)
        diag_cb = np.diag(flipped_dct_cb, k=k_diag)
        diag_cr = np.diag(flipped_dct_cr, k=k_diag)
        if flip:
            diag_y = diag_y[::-1]
            diag_cb = diag_cb[::-1]
            diag_cr = diag_cr[::-1]
        dct_y_zigzag.append(diag_y)
        dct_cb_zigzag.append(diag_cb)
        dct_cr_zigzag.append(diag_cr)
        flip = not flip
    res = np.concatenate(
        [np.concatenate(dct_y_zigzag), np.concatenate(dct_cb_zigzag), np.concatenate(dct_cr_zigzag)])
    return res


def main():
    file = "./dataset/tests/1.jpg"
    features = color_layout_desc(file, rows=2, cols=2)
    features = color_layout_desc("./dataset/tests/2.png", rows=2, cols=2)
    # print(dominant_color_desc(file))
    # print(color_mean_desc(file))
    # print(color_variance_desc(file))


if __name__ == "__main__":
    main()
