from pathlib import Path
import shutil
from tqdm import tqdm

from util.consts import FILE_CAPTIONS_ORIG


def main(in_path, out_path):
    in_path = Path(in_path)
    out_path = Path(out_path)

    if in_path.name.endswith('.srt'):
        captions_paths = [in_path]
        out_paths = [out_path/in_path.stem/FILE_CAPTIONS_ORIG]

    else:
        with in_path.open('r') as f:
            captions_paths = [Path(l.strip()) for l in f if l.strip()]
        out_paths = [out_path/p.stem/FILE_CAPTIONS_ORIG for p in captions_paths]

    # Prune videos that have existing captions
    msg = []
    for i in range(len(captions_paths) - 1, -1, -1):
        if out_paths[i].exists():
            msg.append("Skipping copying captions for video '{}': '{}' "
                       "already exists.".format(captions_paths[i].stem, out_paths[i]))
            captions_paths.pop(i)
            out_paths.pop(i)

    if not captions_paths:
        print('All videos have existing original captions.')
        return

    if msg:
        print(*msg, sep='\n')

    for src, dest in zip(tqdm(captions_paths, desc='Copying original captions',
        unit='video'), out_paths
    ):
        shutil.copy(src, dest)
