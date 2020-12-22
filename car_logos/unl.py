import cv2
import numpy as np

from simple_shape_descriptors import object_contour, prepare_dataset
from signature import center_of_contour, calculate_dists


def unl(img):
    """
    UNL transformation

    Returns 2D image with polar coordinates
    """
    # Moje
    contour = object_contour(img)
    x, y = center_of_contour(img, contour)
    # thresh = cv2.Canny(img, 30, 200)
    polar_image = cv2.linearPolar(img, center=(x, y), maxRadius=np.max(img.shape), flags=cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)

    cv2.imshow("Polar image", polar_image)

    img_fft = np.fft.fft2(polar_image)
    # spectrum = np.log(1 + np.abs(img_fft))
    import matplotlib.pyplot as plt
    zeroed = img_fft.copy()
    zeroed[10:, 10:] = 0
    inverse = np.fft.ifft2(zeroed)
    plt.imshow(np.abs(inverse), "gray")
    plt.title("Spectrum")
    plt.show()

    # out = []
    # for i in range(0, size):
    #     tmp = []
    #     for j in range(0, size):
    #         tmp.append(spectrum[i][j])
    #     out.append(tmp)
    # return list(np.concatenate(out).flat)


    # cv2.cartToPolar(contours_x, contours_y)
    # dists = calculate_dists(contour, [x, y])
    # downsampled_y, downsampled_x = scale_dists_length(dists, length)

    # return downsampled_y, downsampled_x
    cv2.waitKey()


def main():
    X_train, y_train, X_test, y_test = prepare_dataset()

    img = cv2.imread(X_train[2], cv2.IMREAD_GRAYSCALE).astype("uint8")
    cv2.imshow("Original image", img)
    unl(img)


if __name__ == "__main__":
    main()
