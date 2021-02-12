def unl_with_theta(img):
    """
    Porzucone podejście, ale może się kiedyś przyda.
    """
    contour = object_contour(img)

    x, y = center_of_contour(img, contour)
    dists = calculate_dists(contour, [x, y])
    dists = dists/np.max(dists)
    theta = np.arctan2(contour[:, 0, 1]-y, contour[:, 0, 0]-x)

    # Problem jest w tym, że to jest plot! :D
    # A my w sumie chcemy 2d obrazek, żeby na nim zrobić 2d fourier

    # plt.plot(theta, dists)
    plt.plot(dists)
    plt.plot(theta)
    
    plt.show()
    print("theta")
    print(theta)
    print("dists")
    print(dists)

