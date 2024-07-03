from moviepy.editor import VideoFileClip
def convert(input, output):
    video = VideoFileClip(input)
    audio = video.audio
    audio.write_audiofile(output, codec='pcm_s16le')
    audio.close()
    video.close()