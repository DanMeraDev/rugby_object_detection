from pytube import YouTube

url = "https://www.youtube.com/watch?v=OOu7w9UVuXs"

yt = YouTube(url)

video = yt.streams.get_highest_resolution()

video.download()

print(f"Video '{yt.title}' descargado correctamente.")
