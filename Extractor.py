import os
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import yt_dlp
import librosa
import numpy as np
import matplotlib.pyplot as plt

class Extractor(object):

    def __init__(self, url) -> None:
        self.video_url = url
        self.video_info = {}
        # load firefox profile with UBlock extension installed
        # source : https://www.lambdatest.com/blog/adding-firefox-extensions-with-selenium-in-python/

    def reading_audio(self, filename):
        print("[START]","Reading Audio...")
        y, sr = librosa.load(filename) # raw data / sample rate
        print("[INFO]", "sample rate", sr) # 22050 samples per second
        print("[INFO]", y.shape) # audio_duration (in secs) * sr 

        print("[INFO]", librosa.get_duration(y=y, sr=sr), "seconds")
        print("[INFO]", int(librosa.get_duration(y=y, sr=sr) / 60), "minutes")
        print("[END]","Reading Audio...")

        return y, sr
    
    def get_rms(self, y, sr, frame_length, hop_length, k):
        print("[START]","Calculate RMS...")
        # spectogram
        stft = librosa.stft(y=y, n_fft=frame_length, hop_length=hop_length)
        print(stft)
        # rms
        rms = librosa.feature.rms(S=stft, frame_length=frame_length, hop_length=hop_length)
        print(rms)
        
        thresh = rms.mean() + rms.std() * k

        frames = range(len(rms))
        t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        
        fig = plt.figure()

        ax1 = fig.add_subplot(2, 1, 1)  # 2 rows, 1 column, first plot
        ax2 = fig.add_subplot(2, 1, 2)  # 2 rows, 1 column, second plot

        ax1.hist(rms)
        ax2.plot(t, rms/rms.max(), 'r--')
        ax2.axhline(y=thresh/rms.max(), color='g', linestyle='--')
        librosa.display.waveshow(y, sr=sr, alpha=0.4)

        plt.savefig("audio_analysis_spectro.jpg")
        print("[END]","Calculate RMS...")

    def get_short_term_energy(self, y, sr, frame_length, hop_length, k):
        print("[START]","Calculate Short Term Energy...")
        
        energy = np.array([
            sum(abs(y[i:i+frame_length]**2))
            for i in range(0, len(y), hop_length)
        ])

        print("[INFO]", len(energy))
        print("[INFO]", "max_energy", energy.max())
        print("[INFO]", "mean_energy", energy.mean())
        print("[INFO]", "std_energy", energy.std())
        thresh = energy.mean() + energy.std() * k

        frames = range(len(energy))
        t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        
        fig = plt.figure()

        ax1 = fig.add_subplot(2, 1, 1)  # 2 rows, 1 column, first plot
        ax2 = fig.add_subplot(2, 1, 2)  # 2 rows, 1 column, second plot

        ax1.hist(energy)
        ax2.plot(t, energy/energy.max(), 'r--')
        ax2.axhline(y=thresh/energy.max(), color='g', linestyle='--')
        librosa.display.waveshow(y, sr=sr, alpha=0.4)

        plt.savefig("audio_analysis.jpg")
        print("[END]","Calculate Short Term Energy...")

        return energy
    
    def get_threshold(self, energy, k):
        M = energy.mean()
        S = energy.std()
        thresh = M + S*k

        return thresh

    def get_moments(self, energy, thresh, sr, hop_length):
        print("[START]","Get Moments...")
        print("[INFO]","Threshold", thresh)

        start_moments =  []
        end_moments = []
        start = -1
        end = -1
        temp = -1

        for i, val in enumerate(energy):
            if val >= thresh:
                if start == -1:
                    start = i
                temp = val
            else:
                if temp >= thresh:
                    end = i-1
                    start_moments.append(start)
                    end_moments.append(end)
                    temp = -1
                    start = -1
                    end = -1

        s_t = librosa.frames_to_time(frames=start_moments,sr=sr, hop_length=hop_length)
        e_t = librosa.frames_to_time(frames=end_moments, sr=sr, hop_length=hop_length)

        print(["INFO"], start_moments)
        print(["INFO"], s_t)
        print(["INFO"], end_moments)
        print(["INFO"], e_t)

        return s_t, e_t
    
    def download_moments(self, folder, start_moments, end_moments):
        i = 1
        for start_time, end_time in zip(start_moments, end_moments):
            if end_time - start_time > 2:
                ffmpeg_extract_subclip('./download/'+folder+'/video.mp4', int(start_time), int(end_time), targetname='./download/'+folder+"/"+ "highlight_" + str(i) +".mp4")
                i += 1

        
    def download_audio(self, folder):
        print("[START]","Downloading Audio...")

        ydl_opts = {
            'format': 'ba[ext=m4a]',
            'outtmpl': './download/'+folder+'/audio.%(ext)s'
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.video_info = ydl.extract_info(self.video_url)
            print("[END]","Downloading Audio...")
        except yt_dlp.utils.DownloadError as e:
            print("[ERROR]", e.exc_info)
            print("[END]","Downloading Audio...")
            return False
        return True

    
    def download_video(self, folder):
        print("[START]","Downloading Video...")

        ydl_opts = {
            'format': 'bv[ext=mp4]',
            'outtmpl': './download/'+folder+'/video.%(ext)s'
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.video_info = ydl.extract_info(self.video_url)
            print("[END]","Downloading Video...")
        except yt_dlp.utils.DownloadError as e:
            print("[ERROR]", e.exc_info)
            print("[END]","Downloading Video...")
            return False
        return True