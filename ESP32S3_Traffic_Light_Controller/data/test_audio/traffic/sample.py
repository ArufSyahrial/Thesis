import os
import wave
import pylab
import struct

def graph_spectrogram(wav_file, figure_name):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(0.84, 0.84))
    pylab.subplot(111)
    pylab.axis('off')
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(figure_name, transparent=True)

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

# Process all .wav files in the folder
def process_all_wav_files(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):  # Check for .wav files
            wav_file = os.path.join(folder, filename)  # Full path to the .wav file
            figure_name = os.path.join(folder, f"{os.path.splitext(filename)[0]}_spectrogram.png")
            print(f"Processing {filename}...")
            graph_spectrogram(wav_file, figure_name)
            print(f"Spectrogram saved as {figure_name}\n")

if __name__ == '__main__':
    folder = '.'  # Default to the current directory
    process_all_wav_files(folder)
