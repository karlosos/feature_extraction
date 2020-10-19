# http://speech.zone/autocorrelation/
# https://gist.github.com/endolith/255291

from scipy.signal import correlate, find_peaks
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np


def find_f0_autocorelation(signal, fs):
    # transformation to mono
    if signal.ndim == 2:
        signal = signal[:, 0]

    signal = signal[:1000]

    # autocorelation
    corr = correlate(signal, signal, mode='full')
    corr = corr[corr.size//2::]

    # finding peaks
    peaks, peaks_properties = find_peaks(corr, height=0)
    peak = peaks[np.argmax(peaks_properties['peak_heights'])]
    f0 = fs/peak
    print(f0)

    return f0, corr, peaks


if __name__ == "__main__":
    signal, fs = sf.read("lab_2/a_C3_ep44.wav")

    f0, corr, peaks = find_f0_autocorelation(signal, fs)

    fig, (ax_orig, ax_corr) = plt.subplots(2, 1)
    ax_orig.plot(signal)
    ax_orig.set_title('Original signal')

    ax_corr.plot(corr)
    ax_corr.plot(peaks, corr[peaks], "x")
    ax_corr.set_title('Correlation')
    fig.tight_layout()
    plt.show()
