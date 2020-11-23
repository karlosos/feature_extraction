import cv2
import numpy as np
import os
from pathlib import Path

folders = ["01_bmw", "01_chevrolet", "02_kia", "03_mitsubishi", "04_opel",
           "05_peugeout", "06_skoda", "07_subaru", "07_honda", "08_tesla",
           "09_toyota", "09_renault", "10_volkswagen", "10_volvo"]

for name in folders:
    files = os.listdir('./dataset/'+name)
    for f in files:
        filename = "./dataset/" + name + "/" + f
        im_in = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        im_in = cv2.copyMakeBorder(im_in, 20, 20, 20, 20,
                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Threshold.
        # Set values equal to or above 220 to 0.
        # Set values below 220 to 255.
        th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
        im_th = cv2.adaptiveThreshold(
            im_in, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # im_th = cv2.Canny(im_in, 100, 200)

        # Copy the thresholded image.
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv
        kernel = np.ones((5, 5), np.uint8)
        im_out = cv2.morphologyEx(im_out, cv2.MORPH_OPEN, kernel)
        im_out = cv2.morphologyEx(im_out, cv2.MORPH_CLOSE, kernel)

        # Display images.
        # save binarized
        if not os.path.exists("./dataset/binarized/" + name):
            os.makedirs("./dataset/binarized/" + name)
        path_img_target = os.path.join(
            "./dataset/binarized/" + name + "/" + f)
        path_img_target = str(Path(path_img_target).with_suffix('.png'))
        print(path_img_target)
        cv2.imwrite(path_img_target, im_out)
