import subprocess

videos = subprocess.check_output(['gsutil', 'ls', 'gs://esper/tvnews/videos/']).decode("utf-8")
videos = videos.split()
print(videos[:10])
count = 0
for video in videos[1:]:
	date = int(video.split("_")[1])
	if date > 20190723:
		count += 1
print(count)
