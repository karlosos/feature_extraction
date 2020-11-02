import librosa
import numpy as np
import pyaudio
import wave
import struct

NAMES = ['c4', 'c#4', 'd4', 'd#4', 'e4', 'f4', 'f#4', 'g4', 'g#4', 'a4', 'a#4', 'b4',
            'c5', 'c#5', 'd5', 'd#5', 'e5', 'f5', 'f#5', 'g5', 'g#5', 'a5', 'a#5', 'b5',
            'c6']

FREQS = [261.626, 277.182, 293.664, 311.126, 329.627, 349.228, 369.994, 391.995, 415.304, 440, 466.163, 493.883,
        523.251, 554.365, 587.329, 622.253, 659.255, 698.456, 739.988, 783.990, 830.609, 880, 932.327, 987.766,
        1046.502]


def detect_pitch(magnitudes, pitches, t):
    index = magnitudes[:, t].argmax()
    pitch = pitches[index, t]

    return pitch


def fit_notes(note_frequencies):
    """
    Dopasowanie częstotliwości do ustalonych częstotliwości dyskretnych oraz ich nazw
    """
    notes_f = []
    notes_names = []

    for f in note_frequencies:
        # find closest frequency
        frequency = min(FREQS, key=lambda x: abs(x - f))
        notes_f.append(frequency)
        notes_names.append(NAMES[FREQS.index(frequency)])

    return notes_f, notes_names 


if __name__ == "__main__":
    filenames=['1 WlazlKotek (glos m).wav', 
                '2 WlazlKotek (glos m).wav',
                '3 WlazlKotek (glos m nisko).wav',
                '4 WlazlKotek (glos m nisko).wav',
                '5 WlazlKotek (pianino).wav',
                '6 WlazlKotek (pianino).wav',
                '7 WlazlKotek (dziecko).wav',
                '8 WlazlKotek (dziecko).wav',
                '9 WlazlKotek (pianino).wav',
                '10 WlazlKotek (youtube).wav',
                '11 WlazlKotek (youtube).wav']
    
    for filename in filenames:
        print("=========")
        print(filename)
        y, sr = librosa.load("lab_4/"+filename, sr=44000)
        # https://librosa.org/doc/latest/generated/librosa.piptrack.html?highlight=piptrack#librosa.piptrack
        # Pitch tracking on thresholded parabolically-interpolated STFT
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=1600)

        f = []
        print(magnitudes.shape[1])
        for i in range(0, magnitudes.shape[1], magnitudes.shape[1]//15):
            pitch = detect_pitch(magnitudes, pitches, i)
            if pitch < 300:
                continue
            elif pitch > 700:
                pitch /= 2
            f.append(pitch)

        notes, names = fit_notes(f)
        print(names)
        print(notes)


        """
        WYNIKI:
            (.venv) C:\Dev\feature_extraction> cd c:\Dev\feature_extraction && cmd /C "c:\Dev\feature_extraction\.venv\Scripts\python.exe c:\Users\karol\.vscode\extensions\ms-python.python-2020.10.332292344\pythonFiles\lib\python\debugpy\launcher 11649 -- c:\Dev\feature_extraction\lab_4\lab4.py "
            =========
            1 WlazlKotek (glos m).wav
            469
            ['g4', 'c#5', 'a#4', 'a#4', 'd#5', 'g4', 'g#4', 'c5', 'c5', 'c#5']
            [391.995, 554.365, 466.163, 466.163, 622.253, 391.995, 415.304, 523.251, 523.251, 554.365]
            =========
            2 WlazlKotek (glos m).wav
            679
            ['c5', 'b4', 'a#4', 'a#4', 'd#5', 'c5', 'g#4', 'f4', 'c#5', 'c5']
            [523.251, 493.883, 466.163, 466.163, 622.253, 523.251, 415.304, 349.228, 554.365, 523.251]
            =========
            3 WlazlKotek (glos m nisko).wav
            544
            ['d5', 'c5', 'g#4', 'c#5', 'a#4', 'a#4', 'd#4', 'd#5', 'd5']
            [587.329, 523.251, 415.304, 554.365, 466.163, 466.163, 311.126, 622.253, 587.329]
            =========
            4 WlazlKotek (glos m nisko).wav
            696
            ['a#4', 'd5', 'g#4', 'd#4', 'd5', 'a#4']
            [466.163, 587.329, 415.304, 311.126, 587.329, 466.163]
            =========
            5 WlazlKotek (pianino).wav
            1112
            ['c#5', 'a#4', 'a#4', 'b4', 'g4', 'f4', 'f#4', 'a#4', 'c#5', 'c#5', 'c#5']
            [554.365, 466.163, 466.163, 493.883, 391.995, 349.228, 369.994, 466.163, 554.365, 554.365, 554.365]
            =========
            6 WlazlKotek (pianino).wav
            987
            ['c#5', 'a#4', 'a#4', 'b4', 'g#4', 'g#4', 'g#4', 'f#4', 'a#4', 'a#4', 'c#5', 'c#5', 'c#5', 'c#5']
            [554.365, 466.163, 466.163, 493.883, 415.304, 415.304, 415.304, 369.994, 466.163, 466.163, 554.365, 554.365, 554.365, 554.365]
            =========
            7 WlazlKotek (dziecko).wav
            502
            ['c5', 'f4', 'g4', 'g4', 'f4', 'd#4', 'd5']
            [523.251, 349.228, 391.995, 391.995, 349.228, 311.126, 587.329]
            =========
            8 WlazlKotek (dziecko).wav
            372
            ['d#5', 'd4', 'd5', 'd#4', 'd5', 'e4', 'g4']
            [622.253, 293.664, 587.329, 311.126, 587.329, 329.627, 391.995]
            =========
            9 WlazlKotek (pianino).wav
            1096
            ['c#5', 'a#4', 'a#4', 'b4', 'g#4', 'g#4', 'g#4', 'f#4', 'a#4', 'c#5', 'c#5', 'c#5']
            [554.365, 466.163, 466.163, 493.883, 415.304, 415.304, 415.304, 369.994, 466.163, 554.365, 554.365, 554.365]
            =========
            10 WlazlKotek (youtube).wav
            321
            ['c#5', 'd#4']
            [554.365, 311.126]
            =========
            11 WlazlKotek (youtube).wav
            276
            ['e5', 'e4', 'e4', 'c#5', 'd5', 'e4']
            [659.255, 329.627, 329.627, 554.365, 587.329, 329.627]
        """
