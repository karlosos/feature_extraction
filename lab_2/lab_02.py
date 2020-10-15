# http://speech.zone/autocorrelation/
# https://gist.github.com/endolith/255291

from scipy.signal import correlate, find_peaks
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    signal, fs = sf.read("lab_2/a_C4_ugp44.wav")

    # transform to mono
    if signal.ndim == 2:
        signal = signal[:, 0]

    signal = signal[:fs//2]

    print(signal.shape, fs)
    corr = correlate(signal, signal, mode='full')
    corr = corr[corr.size//2::]
    peaks, _ = find_peaks(corr, height=0)

    time = corr.shape[0]
    fig, (ax_orig, ax_corr) = plt.subplots(2, 1)
    ax_orig.plot(signal)
    ax_orig.set_title('Original signal')

    ax_corr.plot(corr)
    ax_corr.plot(peaks, corr[peaks], "x")
    ax_corr.set_title('Correlation')
    fig.tight_layout()
    plt.show()
    f0 = fs/peaks[0]
    print(f0)
