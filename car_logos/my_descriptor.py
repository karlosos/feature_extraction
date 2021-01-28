import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def my_desc(filename, size):
    # Przekształcenie do skali szarosci
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    img = cv2.bitwise_not(img)

    # Przekształcenie do współrzędnych biegunowych obrazu po wektoryzacji
    max_radius = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    moment = cv2.moments(img)
    x = int(moment["m10"] / moment["m00"])
    y = int(moment["m01"] / moment["m00"])
    centroid = (x, y)

    polar_image = cv2.linearPolar(
        img, centroid, max_radius, cv2.WARP_FILL_OUTLIERS
    )
    polar_image = polar_image.astype(np.uint8)
    img_dct = dct(dct(polar_image.T, norm='ortho').T, norm='ortho')
    dct_values = zigzag(img_dct)[:size]

    return dct_values


def zigzag(a):
    """
    Create zigzag vector from matrix
    :param a: input matrix
    :returns: vector with length m*n with matrix values under zigzag indexing
    """
    m, n = a.shape
    solution = [[] for i in range(m + n - 1)]

    for i in range(m):
        for j in range(n):
            sum = i + j
            value = a[i][j]
            index = (i, j)
            if sum % 2 == 0:
                solution[sum].insert(0, (value, index))
            else:
                solution[sum].append((value, index))

    output_vector = []
    for i in solution:
        for j in i:
            output_vector.append(j[0])

    return output_vector 


def my_desc_plagiat(filename, size):
    # Przekształcenie do skali szarosci
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    img = cv2.bitwise_not(img)
    plt.imshow(img, cmap="gray")
    plt.title("Obraz po skalowaniu")
    plt.show()

    # Wektoryzacja KNN
    height, width = img.shape
    kmeans = KMeans(n_clusters=10, random_state=0).fit(img.reshape(-1, 1))
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    img_quant = centers[labels, :].reshape(height, width).astype("uint8")


    plt.imshow(img_quant, cmap="gray")
    plt.title("Obraz po kwantyzacji")
    plt.show()

    # Przekształcenie do współrzędnych biegunowych obrazu po wektoryzacji
    max_radius = np.sqrt(((img_quant.shape[0] / 2.0) ** 2.0) + ((img_quant.shape[1] / 2.0) ** 2.0))
    moment = cv2.moments(img_quant)
    x = int(moment["m10"] / moment["m00"])
    y = int(moment["m01"] / moment["m00"])
    centroid = (x, y)
    polar_image = cv2.linearPolar(
        img_quant, centroid, max_radius, cv2.WARP_FILL_OUTLIERS
    )
    polar_image = polar_image.astype(np.uint8)
    

    plt.imshow(polar_image, cmap="gray")
    plt.title("Obraz w współrzędnych biegunowych")
    plt.show()


    return polar_image.sum(axis=0).tolist() + polar_image.sum(axis=1).tolist()

def my_desc_old_2(filename, size):
    # Przekształcenie do skali szarosci
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    img = cv2.bitwise_not(img)

    # Wektoryzacja KNN
    height, width = img.shape
    kmeans = KMeans(n_clusters=10, random_state=0).fit(img.reshape(-1, 1))
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    img_quant = centers[labels, :].reshape(height, width).astype("uint8")

    cluster_values = np.unique(img_quant)

    plt.imshow(img_quant, cmap="gray")
    plt.title("Obraz po kwantyzacji")
    plt.show()

    # Przekształcenie do współrzędnych biegunowych obrazu po wektoryzacji
    max_radius = np.sqrt(((img_quant.shape[0] / 2.0) ** 2.0) + ((img_quant.shape[1] / 2.0) ** 2.0))
    moment = cv2.moments(img_quant)
    x = int(moment["m10"] / moment["m00"])
    y = int(moment["m01"] / moment["m00"])
    centroid = (x, y)
    polar_image = cv2.linearPolar(
        img_quant, centroid, max_radius, cv2.WARP_FILL_OUTLIERS
    )
    polar_image = polar_image.astype(np.uint8)

    plt.imshow(polar_image, cmap="gray")
    plt.title("Obraz w współrzędnych biegunowych")
    plt.show()

    # Fourier 2D
    # Przekształcenie fouriera
    # img_fft = np.fft.fft2(polar_image)
    # spectrum = np.log(1 + np.abs(img_fft))
    img_dct = dct(dct(img.T, norm='ortho').T, norm='ortho')
    
    # plt.imshow(spectrum, cmap="gray")
    # plt.title("Widmo 2D Fourier")
    # plt.show()

    # Wycinek fouriera
    fourier_values = list(img_dct[:size, :size].flat)

    res = fourier_values + cluster_values.tolist()
    res = fourier_values

    return res 

def my_desc_old(filename, size):
    # Przekształcenie do skali szarosci
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (126, 128), interpolation=cv2.INTER_LINEAR)
    img = cv2.bitwise_not(img)

    # Wektoryzacja KNN
    height, width = img.shape
    kmeans = KMeans(n_clusters=10, random_state=0).fit(img.reshape(-1, 1))
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    img_quant = centers[labels, :].reshape(height, width).astype("uint8")

    cluster_values = np.unique(img_quant)

    # plt.imshow(img_quant, cmap="gray")
    # plt.title("Obraz po kwantyzacji")
    # plt.show()

    # Przekształcenie do współrzędnych biegunowych obrazu po wektoryzacji
    max_radius = np.sqrt(((img_quant.shape[0] / 2.0) ** 2.0) + ((img_quant.shape[1] / 2.0) ** 2.0))
    moment = cv2.moments(img_quant)
    x = int(moment["m10"] / moment["m00"])
    y = int(moment["m01"] / moment["m00"])
    centroid = (x, y)
    polar_image = cv2.linearPolar(
        img_quant, centroid, max_radius, cv2.WARP_FILL_OUTLIERS
    )
    polar_image = polar_image.astype(np.uint8)

    # plt.imshow(polar_image, cmap="gray")
    # plt.title("Obraz w współrzędnych biegunowych")
    # plt.show()

    # Fourier 2D
    # Przekształcenie fouriera
    img_fft = np.fft.fft2(polar_image)
    spectrum = np.log(1 + np.abs(img_fft))
    
    # plt.imshow(spectrum, cmap="gray")
    # plt.title("Widmo 2D Fourier")
    # plt.show()

    # Wycinek fouriera
    img_fft = np.fft.fft2(polar_image)
    spectrum = np.log(1 + np.abs(img_fft))
    fourier_values = list(spectrum[:size, :size].flat)

    res = fourier_values + cluster_values.tolist()
    res = fourier_values

    return res 

def my_desc_with_images(filename, size):
    # Przekształcenie do skali szarosci
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    img = cv2.bitwise_not(img)

    # plt.figure("Polar DCT Greyscale Descriptor", figsize=(16, 3))

    # # ax1 = plt.subplot(1, 4, 1)
    # plt.imshow(img, cmap="gray")
    # # plt.title("Obraz po skalowaniu")
    # plt.show()

    # Przekształcenie do współrzędnych biegunowych obrazu po wektoryzacji
    max_radius = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    moment = cv2.moments(img)
    x = int(moment["m10"] / moment["m00"])
    y = int(moment["m01"] / moment["m00"])
    centroid = (x, y)

    # from matplotlib.patches import Circle
    # fig, ax = plt.subplots(1)
    # ax.imshow(img, cmap="gray")
    # circ = Circle((x, y), 2)
    # ax.add_patch(circ)
    # plt.show()



    polar_image = cv2.linearPolar(
        img, centroid, max_radius, cv2.WARP_FILL_OUTLIERS
    )
    polar_image = polar_image.astype(np.uint8)

    # # plt.subplot(1, 4, 2, sharey=ax1)
    # plt.imshow(polar_image, cmap="gray")
    # # plt.title("Obraz po transformacji do wsp. biegunowych")
    # plt.show()
    
    img_dct = dct(dct(polar_image.T, norm='ortho').T, norm='ortho')

    # # plt.subplot(1, 4, 3, sharey=ax1)
    # plt.imshow(img_dct, cmap="gray")
    # # plt.title("Obraz po transformacie DCT")
    # plt.show()

    dct_values = zigzag(img_dct)[:32]

    # # ax4 = plt.subplot(1, 4, 4, sharey=ax1)
    # ax4 = plt.subplot(1, 1, 1)
    # plt.imshow(np.array(dct_values).reshape(-1, 1), cmap="gray")
    # # plt.title("Wektor wynikowy")
    # # plt.setp(ax4.get_xticklabels(), visible=False)
    # plt.setp(ax4.get_xticklabels(), visible=False)
    # plt.show()

    return dct_values
