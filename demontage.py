from PIL import Image
import os
import glob
import numpy as np
import tqdm

def demontage(path):
	im = Image.open(path)
	xes = list(enumerate(range(25, im.size[0], 250)))
	n_cols = len(xes)
	for xi, x in xes:
		for yi, y in enumerate(range(25, im.size[1], 250)):
			im2 = im.crop((x, y, x+200, y+200))
			im2.save("minis/{:02d}.{:03d}.png".format(i, yi * n_cols + xi))

def montage(input_paths, width, height):
	out_path = "{:02d}x{:02d}".format(width, height)
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	chunk = width * height
	for i in tqdm.tqdm(range(0, len(input_paths), chunk)):

		# Load a chunk of input images
		inputs = [[np.zeros((200, 200, 3)) for _ in range(width)] for _ in range(height)]
		for j in range(chunk):
			if i + j < len(input_paths):
				inputs[j // width][j % width] = np.array(Image.open(input_paths[i + j]))

		# Stitch them together
		stitched = np.vstack([np.hstack(row) for row in inputs])
		stitched = np.uint8(stitched)

		# Save them
		Image.fromarray(stitched, 'RGB').save(os.path.join(out_path, "{:03d}.png".format(i)))


if __name__ == '__main__':
	# for i in range(27):
	# 	demontage("CNNW_20200124_140000_CNN_Newsroom_With_Poppy_Harlow_and_Jim_Sciutto/{}.png".format(i))
	
	for width in range(1, 11):
		for height in range(1, 11):
			montage(sorted(glob.glob('minis/*')), width, height)


