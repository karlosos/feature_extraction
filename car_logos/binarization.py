import os
from skimage.color import rgb2gray, rgba2rgb
from skimage import img_as_ubyte
from skimage import io
from skimage.filters import threshold_otsu
from skimage.feature import canny
from scipy import ndimage as ndi
import numpy as np

folders = ["01_bmw", "01_chevrolet", "02_kia", "03_mitsubishi", "04_opel",
           "05_peugeout", "06_skoda", "07_subaru", "07_honda", "08_tesla",
           "09_toyota", "09_renault", "10_volkswagen", "10_volvo"]

for name in folders:
    for i in range(1, 11):
        filename = "./dataset/" + name + "/" + str(i) + ".jpg"
        img = io.imread(filename)
        img2 = rgb2gray(img)
        # thresh = threshold_otsu(img2)
        # binary = img2 > thresh
        # binary = img_as_ubyte(binary)
        edges = canny(img2)
        binary = ndi.binary_fill_holes(edges, structure=np.ones((20, 20)))

        # save binarized
        if not os.path.exists("./dataset/binarized/" + name):
            os.makedirs("./dataset/binarized/" + name)
        path_img_target = os.path.join("./dataset/binarized/" + name + "/" + str(i) + ".png")
        io.imsave(path_img_target, binary)
