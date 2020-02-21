"""
This is a user friendly script to label images as one of the 6 democratic
candidates in the January 14th debate. The labels in ground_truth_labels.txt
are for the first 600 images of that debate. The source file is named
gs://esper/tvnews/videos/CNNW_20200115_020000_CNN_Democratic_Debate.mp4
"""
import glob
import subprocess
import os
import sys, tty, termios

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

keys = ['E', 'J', 'B', 'P', 'A', 'T']
options = ['Elizabeth Warren', 'Joe Biden', 'Bernie Sanders', 'Pete Buttigieg',
           'Amy Klobuchar', 'Tom Steyer']
mapping = dict(zip(keys, options))

if __name__ == '__main__':
	images = glob.glob('minis/*')
	# assert not os.path.exists('ground_truth_labels.txt')
	old_labels = open('ground_truth_labels.txt').read().split("\n")
	print(old_labels)
	with open('ground_truth_labels.txt', 'w') as f:
		for i, image in enumerate(list(sorted(images))):
			if i < len(old_labels) - 1 and old_labels[i] != 'Abby Phillip':
				label = old_labels[i]
			else:
				subprocess.run(['imgcat', image])
				print("{} of {}".format(i, len(images)))
				print("Who is this?")
				for key, option in zip(keys, options):
					print("[{}] {}".format(key, option))
				print("[Space] Other")
				inp = getch().upper()
				label = mapping[inp] if inp in mapping else ""
				print(label)
			f.write(label + "\n")
			f.flush()
