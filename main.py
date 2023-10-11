from Extractor import Extractor
from datetime import datetime

if __name__ == "__main__":
    folder = datetime.now().strftime("%Y-%b-%d_%H-%M-%S")

    yt_video_url = "https://www.youtube.com/watch?v=3OFj6l2tQ9s" # mr beast
    yt_video_url = "https://www.youtube.com/watch?v=-FZdSrqmrrk" # football 1
    yt_video_url = "https://www.youtube.com/watch?v=6CIMX28_f4Q" # football 2

    extractor = Extractor(yt_video_url)

    extractor.download_audio(folder)

    y, sr = extractor.reading_audio(f'./download/{folder}/audio.m4a')

    frame_length = sr * 5 # 5 seconds
    hop_length = frame_length // 4 
    k = 1.15 # coefficient of threshold formula

    energy = extractor.get_short_term_energy(y, sr, frame_length, hop_length, k)
    threshold_energy = extractor.get_threshold(energy, k)
    
    s, e = extractor.get_moments(energy, threshold_energy, sr, hop_length)
 
    extractor.download_video(folder)
    extractor.download_moments(folder, s, e)