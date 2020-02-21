import cv2
import io
import json

import numpy as np
from PIL import Image

from consts import OUTDIR_MONTAGES
from config import MONTAGE_WIDTH, MONTAGE_HEIGHT
from utils import get_base_name

IMG_SIZE = 200
BLOCK_SIZE = 250
HALF_BLOCK_SIZE = BLOCK_SIZE / 2
PADDING = int((BLOCK_SIZE - IMG_SIZE) / 2)
assert IMG_SIZE + 2 * PADDING == BLOCK_SIZE, \
    'Bad padding: {}'.format(IMG_SIZE + 2 * PADDING)

BLANK_IMAGE = np.zeros((BLOCK_SIZE, BLOCK_SIZE, 3), dtype=np.uint8)

def create_montage_bytes(img_filepaths, nrows=MONTAGE_HEIGHT,
                         ncols=MONTAGE_WIDTH):
    assert len(img_filepaths) <= nrows * ncols
    try:
        ids = [int(get_base_name(p)) for p in img_filepaths]
    except ValueError:
        raise Exception('Face crop filenames must be integer IDs')
    
    stacked_rows = []
    for i in range(0, len(img_filepaths), ncols):
        row_imgs = [convert_image(np.asarray(Image.open(path)))
                for path in img_filepaths[i:i + ncols]]
        row_imgs += [BLANK_IMAGE] * (ncols - len(row_imgs))
        stacked_rows.append(np.hstack(row_imgs))

    stacked_img = np.vstack(stacked_rows)
    stacked_img_meta = {
        'rows': len(stacked_rows),
        'cols': ncols,
        'block_dim': BLOCK_SIZE,
        'content': ids
    }

    with io.BytesIO() as f:
        Image.fromarray(stacked_img).save(f, 'png')
        return f.getvalue(), stacked_img_meta


def convert_image(im):
    im = cv2.resize(im, dsize=(IMG_SIZE, IMG_SIZE))
    im = cv2.copyMakeBorder(
        im, PADDING, PADDING, PADDING, PADDING,
        cv2.BORDER_CONSTANT, value=0)
    return im

