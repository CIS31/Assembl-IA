from moviepy import VideoFileClip

def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from a video file and saves it as an audio file.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the extracted audio file.
    """
    # Load the video file
    video = VideoFileClip(video_path)

    # Extract the audio
    audio = video.audio

    # Write the audio to a file
    audio.write_audiofile(audio_path)

    # Close the video and audio objects
    audio.close()
    video.close()


video_file = "../assets/video1.mp4"
audio_file = "../assets/audio1.wav"

extract_audio_from_video(video_file, audio_file)