import os
from skimage.color import rgb2gray, rgba2rgb
from skimage import io
from skimage.filters import threshold_otsu, threshold_minimum, threshold_triangle, threshold_local, rank, threshold_yen


folders = ["01_bmw", "02_kia", "03_mitsubishi", "04_opel",
           "05_peugeout", "06_skoda", "07_subaru", "08_tesla",
           "09_toyota", "10_volkswagen"]

for name in folders:
    for i in range(0, 10):
        filename = "./dataset/" + name + "/" + str(i) + ".png"
        img = io.imread(filename)
        img2 = rgb2gray(rgba2rgb(img))
        thresh = threshold_triangle(img2)
        binary = img2 > thresh

        path_img_target = os.path.join("./dataset/binarized/" + name + "/" + str(i) + ".png")
        io.imsave(path_img_target, binary)
