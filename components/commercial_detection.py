from multiprocessing import Pool
import os
from pathlib import Path

import pysrt
from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import after, or_pred, overlaps
from tqdm import tqdm

from util.consts import (FILE_BLACK_FRAMES,
                         FILE_CAPTIONS,
                         FILE_METADATA,
                         FILE_COMMERCIALS)
from util.utils import load_json, save_json

BLACK_FRAME_COALESCE_EPSILON = 2
CAPTIONS_COALESCE_EPSILON = 2
RELIABLE_TEXT_DURATION = 5
MIN_COMMERCIAL_TIME = 10
MAX_COMMERCIAL_TIME = 300
COMMERCIAL_FOLD_EPSILON = 5
MIN_LOWERTEXT = 0.5
LOWERCASE_COALESCE_EPSILON = 2
MIN_LOWERWINDOW = 15
MIN_BLANKWINDOW = 30
MAX_BLANKWINDOW = 270
MAX_MERGE_GAP = 120
MAX_MERGE_DURATION = 300


def main(in_path, out_path, force=False):
    video_names = os.listdir(in_path)
    out_paths = [Path(out_path)/name for name in video_names]
    in_path = Path(in_path)

    for p in out_paths:
        p.mkdir(parents=True, exist_ok=True)

    # Prune videos that should not be run
    msg = []
    for i in range(len(video_names) - 1, -1, -1):
        black_frames_path = in_path/video_names[i]/FILE_BLACK_FRAMES
        captions_path = in_path/video_names[i]/FILE_CAPTIONS
        metadata_path = in_path/video_names[i]/FILE_METADATA
        commercials_outpath = out_paths[i]/FILE_COMMERCIALS
        if not black_frames_path.exists():
            msg.append("Skipping commercial detection for video '{}': '{}' "
                       "does not exists.".format(video_names[i], black_frames_path))
        elif not captions_path.exists():
            msg.append("Skipping commercial detection for video '{}': '{}' "
                       "does not exists.".format(video_names[i], captions_path))
        elif not metadata_path.exists():
            msg.append("Skipping commercial detection for video '{}': '{}' "
                       "does not exists.".format(video_names[i], metadata_path))
        elif not force and commercials_outpath.exists():
            msg.append("Skipping commercial detection for video '{}': '{}' "
                       "already exists.".format(video_names[i], commercials_outpath))
        else:
            continue

        video_names.pop(i)
        out_paths.pop(i)

    if not video_names:
        print('All videos have existing detected commercials.')
        return

    if msg:
        print(*msg, sep='\n')


    with Pool() as workers, tqdm(
        total=len(video_names), desc='Detecting commercials', unit='video'
    ) as pbar:
        for video_name, output_dir in zip(video_names, out_paths):
            black_frames_path = in_path/video_name/FILE_BLACK_FRAMES
            captions_path = in_path/video_name/FILE_CAPTIONS
            metadata_path = in_path/video_name/FILE_METADATA
            commercials_outpath = output_dir/FILE_COMMERCIALS

            workers.apply_async(
                process_single,
                args=(str(black_frames_path), str(captions_path),
                      str(metadata_path), str(commercials_outpath)),
                callback=lambda x: pbar.update()
            )

        workers.close()
        workers.join()


def process_single(black_frames_path, captions_path, metadata_path,
                   commercials_outpath):

    # Load original data
    black_frames = load_json(black_frames_path)
    captions = load_captions(captions_path)
    metadata = load_json(metadata_path)

    # Create IntervalSet objects
    black_frames_set = IntervalSet([
        Interval(Bounds3D(
            frame_num / metadata['fps'], (frame_num + 1) / metadata['fps']
        )) for frame_num in black_frames
    ])

    captions_set = IntervalSet([
        Interval(Bounds3D(start, end), payload=text)
        for text, start, end, in captions
    ])

    whole_video = IntervalSet([
        Interval(Bounds3D(0, metadata['frames'] / metadata['fps']))
    ])

    # Detect commercials
    results = detect_commercials(black_frames_set, captions_set, whole_video)

    # Convert commercial intervals back to frames
    results = convert_set_from_seconds_to_frames(results, metadata['fps'])

    # Save results in JSON format
    results = [(r['t1'], r['t2']) for r in results.get_intervals()]
    save_json(results, commercials_outpath)


def detect_commercials(black_frames, captions, whole_video, params=None):
    """
    ---

    Arguments:
        captions (IntervalSet):
        black_frames (IntervalSet):
        whole_video (IntervalSet):
        params (dict):

    Returns:
        IntervalSet of commercials (in seconds).
    """

    black_windows = black_frames.coalesce(
        ('t1', 't2'), Bounds3D.span, epsilon=BLACK_FRAME_COALESCE_EPSILON
    )

    caption_intervals = captions.filter(lambda x: '{' not in x['payload'])

    arrow_intervals = caption_intervals.filter(lambda x: '>>' in x['payload'])
    arrow_announcer_intervals = caption_intervals.filter(
        lambda x: '>> Announcer:' in x['payload']
    )
    arrow_having_intervals = caption_intervals.filter(
        lambda x: '>> HAVING' in x['payload']
    )

    caption_intervals = caption_intervals.coalesce(
        ('t1', 't2'), Bounds3D.span, epsilon=CAPTIONS_COALESCE_EPSILON
    )

    reliable_captions = caption_intervals.filter_size(
        min_size=RELIABLE_TEXT_DURATION
    )

    arrow_intervals = arrow_intervals.minus(
        arrow_announcer_intervals
    ).minus(
        arrow_having_intervals
    ).filter_against(
        reliable_captions,
        predicate=overlaps()
    )

    # Find commercial blocks
    all_blocks = whole_video.minus(black_windows)
    non_commercial_blocks = all_blocks.filter_against(
        arrow_intervals,
        predicate=overlaps()
    )
    commercial_blocks = whole_video.minus(
        non_commercial_blocks.union(black_windows)
    )

    def fold_fn(stack, interval):
        if interval['t2'] - interval['t1'] > MAX_COMMERCIAL_TIME:
            interval = Interval(
                Bounds3D(interval['t1'], interval['t1'] + MAX_COMMERCIAL_TIME)
            )

        if len(stack) == 0:
            stack.append(interval)
        else:
            last = stack.pop()
            if or_pred(overlaps(), after(max_dist=COMMERCIAL_FOLD_EPSILON))(
                interval, last
            ):
                if last['bounds'].span(interval['bounds']).size() \
                        > MAX_COMMERCIAL_TIME:
                    stack.append(Interval(
                        Bounds3D(last['t1'], last['t1'] + MAX_COMMERCIAL_TIME)
                    ))
                else:
                    stack.append(Interval(
                        last['bounds'].span(interval['bounds'])
                    ))
            else:
                stack.append(last)
                stack.append(interval)

        return stack

    commercials = commercial_blocks.fold_to_set(
        fold_fn, init=[]
    ).filter_size(min_size=MIN_COMMERCIAL_TIME)
    commercials_orig = commercials

    def is_lower_text(obj):
        alpha = [c for c in obj['payload'] if c.isalpha()]
        if not alpha:
            return False

        lower = [c for c in alpha if c.islower()]
        return len(lower) / len(alpha) > MIN_LOWERTEXT

    lowercase_intervals = captions.filter(
        is_lower_text
    ).coalesce(
        ('t1', 't2'),
        Bounds3D.span,
        payload_merge_op=lambda a, b: a + ' ' + b,
        epsilon=LOWERCASE_COALESCE_EPSILON
    ).filter_size(min_size=MIN_LOWERWINDOW)

    commercials = commercials.union(lowercase_intervals)

    blank_intervals = whole_video.minus(
        caption_intervals
    ).filter_size(
        min_size=MIN_BLANKWINDOW, max_size=MAX_BLANKWINDOW
    ).minus(
        whole_video.map(lambda x: Interval(Bounds3D(x['t2'] - 60, x['t2'])))
    ).filter_size(min_size=MIN_BLANKWINDOW)

    commercials = commercials.union(blank_intervals)

    commercials = commercials.coalesce(
        ('t1', 't2'), Bounds3D.span, epsilon=MAX_MERGE_GAP
    ).filter_size(
        max_size=MAX_COMMERCIAL_TIME
    ).union(
        commercials_orig
    ).union(
        lowercase_intervals
    ).union(
        blank_intervals
    ).coalesce(
        ('t1', 't2'), Bounds3D.span
    )

    return commercials


def load_captions(path):
    subs = pysrt.open(path)
    captions = [
        (s.text, time_to_second(tuple(s.start)[:4]),
         time_to_second(tuple(s.end)[:4]))
        for s in subs
    ]

    return captions


def convert_set_from_seconds_to_frames(interval_set, fps):
    def map_fn(interval):
        interval_copy = interval.copy()
        curr_bounds = interval['bounds'].copy()
        curr_bounds['t1'] = int(curr_bounds['t1'] * fps)
        curr_bounds['t2'] = int(curr_bounds['t2'] * fps)
        interval_copy['bounds'] = curr_bounds
        return interval_copy

    return interval_set.map(map_fn)


def frame_to_second(frame_num, fps):
    return f / fps


def time_to_second(time):
    seconds = time[0] * 60 * 60 + time[1] * 60 + time[2]
    if len(time) == 4:
        seconds += time[3] / 1000

    return seconds


if __name__ == '__main__':
    main('../sample_output_dir_batch', '../sample_output_dir_batch')
