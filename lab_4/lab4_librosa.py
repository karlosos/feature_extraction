import librosa
import numpy as np
import pyaudio

y, sr = librosa.load("9 WlazlKotek (pianino).wav", sr=40000)
pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=1600)

def detect_pitch(y, sr, t):
    index = magnitudes[:, t].argmax()
    pitch = pitches[index, t]

    return pitch


def play(f, duration):
    p = pyaudio.PyAudio()

    volume = 0.5  # range [0.0, 1.0]
    fs = 44100  # sampling rate, Hz, must be integer
    # duration = 1 # in seconds, may be float
    # f = 440.0  # sine frequency, Hz, may be float
    # generate samples, note conversion to float32 array
    samples = [(np.sin(2 * np.pi * np.arange(fs * duration * 2) * int(f[i]) / fs)).astype(np.float32) for i in range(len(f))]
    print(f'Playing: {f}')
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    # play. May repeat with different volume values (if done interactively)
    [stream.write(volume * sample) for sample in samples]

    stream.stop_stream()
    stream.close()

    p.terminate()


def name_note(f):
    names = ['c4', 'c#4', 'd4', 'd#4', 'e4', 'f4', 'f#4', 'g4', 'g#4', 'a4', 'a#4', 'b4',
             'c5', 'c#5', 'd5', 'd#5', 'e5', 'f5', 'f#5', 'g5', 'g#5', 'a5', 'a#5', 'b5',
             'c6']

    freq = [261.626, 277.182, 293.664, 311.126, 329.627, 349.228, 369.994, 391.995, 415.304, 440, 466.163, 493.883,
            523.251, 554.365, 587.329, 622.253, 659.255, 698.456, 739.988, 783.990, 830.609, 880, 932.327, 987.766,
            1046.502]
    notes = []
    list_names = []
    for item in f:
        frequency = min(freq, key=lambda x: abs(x - item))
        notes.append(frequency)
        list_names.append(names[freq.index(frequency)])

    return notes, list_names


def main():
    f = []
    print(magnitudes.shape[1])
    for i in range(0, magnitudes.shape[1], magnitudes.shape[1]//15):
        pitch = detect_pitch(y, sr, i)
        if pitch < 300:
            continue
        elif pitch > 700:
            pitch /= 2
        f.append(pitch)

    notes, names = name_note(f)
    print(names)
    print(notes)
    play(notes, 1)
if __name__ == "__main__":
    main()