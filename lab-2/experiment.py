import sounddevice as sd
import scipy.io.wavfile as wavfile

fs, audio = wavfile.read("./sounds/sinus.wav")
sd.play(audio, fs)