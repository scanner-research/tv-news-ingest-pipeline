from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import *
from rekall.stdlib import ingest
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat, FlatFormat
from vgrid import SpatialType_Bbox, SpatialType_Caption, Metadata_Generic
from vgrid_jupyter import VGridWidget
import urllib3, requests, os
import math
import pickle

urllib3.disable_warnings()

dev_set = [559, 1791, 3730, 3754, 10323, 11579, 17386, 20689, 24847, 24992, 
           26175, 33800, 40203, 40267, 43637, 50561, 54377, 57990, 59028, 
           63965, 67300]
test_set = [385, 8697, 9215, 9901, 12837, 13993, 14925, 18700, 23541, # 31902,
            32996, 36755, 50164, 52945, 55711, 57748, 59789, 60433, 136732,
            149097, 169420]

VIDEO_COLLECTION_BASEURL = "http://olimar.stanford.edu/hdd/tvnews-commercials"
VIDEO_METADATA_FILENAME = "video_meta_commercials.json"

req = requests.get(os.path.join(VIDEO_COLLECTION_BASEURL, VIDEO_METADATA_FILENAME), verify=False)
video_collection = req.json()

video_metadata = [
    VideoMetadata(v["path"], v["id"], v["fps"], int(v["num_frames"]), v["width"], v["height"])
    for v in video_collection
]

def load_data():
    def load_json(video_baseurl, json_path):
        req = requests.get(os.path.join(video_baseurl, json_path), verify=False)
        json_objs = req.json()
        ism = ingest.ism_from_iterable_with_schema_bounds3D(
            json_objs,
            ingest.getter_accessor,
            {
                'key': 'video_id',
                't1': 'start',
                't2': 'end'
            },
            with_payload = lambda item: item,
            progress = True
        )
        return ism

    COMMERCIALS_JSON = 'all_commercials.json'
    commercials = load_json(VIDEO_COLLECTION_BASEURL, COMMERCIALS_JSON)

    CAPTIONS_PICKLE = "captions_commercials_aligned.pkl"
    req = requests.get(os.path.join(VIDEO_COLLECTION_BASEURL, CAPTIONS_PICKLE), verify=False)
    captions_by_id = pickle.loads(req.content)

    captions = IntervalSetMapping({
        video_id: IntervalSet([
            Interval(Bounds3D(start, end), payload=text)
            for text, start, end in captions_by_id[video_id]
        ])
        for video_id in dev_set + test_set if video_id in captions_by_id
    })

    BLACK_FRAMES_PATH = 'black_frame_all.pkl'
    req = requests.get(os.path.join(VIDEO_COLLECTION_BASEURL, BLACK_FRAMES_PATH), verify=False)

    black_frames_by_id = pickle.loads(req.content)

    black_frames = IntervalSetMapping({
        video_id: IntervalSet([
            Interval(Bounds3D(frame_num, frame_num + 1))
            for frame_num in black_frames_by_id[video_id]
        ])
        for video_id in dev_set + test_set
    })

    whole_video = IntervalSetMapping({
        vm.id: IntervalSet([Interval(Bounds3D(0, vm.num_frames / vm.fps))])
        for vm in video_metadata
    })
    
    return video_metadata, commercials, captions, black_frames, whole_video

vm_by_video = {
    video_id: [vm for vm in video_metadata if vm.id == video_id][0]
    for video_id in dev_set + test_set
}

def frame_second_conversion(c, mode='f2s'):
    def second_to_frame(fps):
        def map_fn(intrvl):
            i2 = intrvl.copy()
            curr_bounds = intrvl['bounds'].copy()
            curr_bounds['t1'] = int(curr_bounds['t1']*float(fps))
            curr_bounds['t2'] = int(curr_bounds['t2']*float(fps))
            i2['bounds'] = curr_bounds
            return i2
        return map_fn
    
    def frame_to_second(fps):
        def map_fn(intrvl):
            i2 = intrvl.copy()
            curr_bounds = intrvl['bounds'].copy()
            curr_bounds['t1'] = curr_bounds['t1']/float(fps)
            curr_bounds['t2'] = curr_bounds['t2']/float(fps)
            i2['bounds'] = curr_bounds
            return i2
        return map_fn
    
    if mode=='f2s':
        fn = frame_to_second
    if mode=='s2f':
        fn = second_to_frame
    output = {}
    for vid, intervals in c.get_grouped_intervals().items():
        output[vid] = intervals.map(fn(vm_by_video[vid].fps))
    return IntervalSetMapping(output)

def frame_to_second_collection(c, cast_to_int = True):
    seconds = frame_second_conversion(c, 'f2s')
    if cast_to_int:
        return seconds.map(lambda intrvl: Interval(
            Bounds3D(int(intrvl['t1']), int(intrvl['t2']))
        ))
    
    return frame_second_conversion(c, 'f2s')

def second_to_frame_collection(c):
    return frame_second_conversion(c, 's2f')

def filter_by_id(ism, valid_ids):
    return IntervalSetMapping({
        vid: ism.get_grouped_intervals()[vid]
        for vid in list(ism.get_grouped_intervals().keys()) if vid in valid_ids
    })

def get_commercial_labels(commercials):
    interval = 10
    segs_dict = {}
    for video_id in dev_set + test_set:
        video = vm_by_video[video_id]
        iset = IntervalSet([
            Interval(Bounds3D(i - interval / 2, i + interval / 2))
            for i in range(0, int(video.num_frames / video.fps), interval)
        ])
        segs_dict[video_id] = iset

    segments = IntervalSetMapping(segs_dict)
    segments_all_negative = segments.map(
        lambda intrvl: Interval(intrvl['bounds'], 0)
    )

    commercial_segments = segments.filter_against(
        commercials, predicate = overlaps()
    ).map(
        lambda intrvl: Interval(intrvl['bounds'], 1)
    )

    commercial_labels = segments_all_negative.minus(
        commercial_segments
    ).union(commercial_segments)
    
    return segments, segments_all_negative, commercial_labels

def evaluate_preds(segments, segments_all_negative, predictions, commercial_labels, video_ids):
    predictions = filter_by_id(predictions, video_ids)
    commercial_labels = filter_by_id(commercial_labels, video_ids)
    
    prediction_segments = segments.filter_against(
        predictions,
        predicate = overlaps()
    ).map(lambda intrvl: Interval(intrvl['bounds'], 1))

    prediction_labels = segments_all_negative.minus(
        prediction_segments
    ).union(prediction_segments)

    prediction_scores = prediction_labels.join(
        commercial_labels,
        predicate = equal(),
        merge_op = lambda i1, i2: Interval(
            i1['bounds'],
            'tp' if i1['payload'] == i2['payload'] and i1['payload'] == 1 else
            'tn' if i1['payload'] == i2['payload'] and i1['payload'] == 0 else
            'fp' if i1['payload'] != i2['payload'] and i1['payload'] == 1 else
            'fn'
        ),
        window = 0
    )
    
    def precision_recall_f1(pred_labels):
        def sum_values(obj):
            return sum([v for v in list(obj.values())])
        tp = sum_values(pred_labels.filter(payload_satisfies(lambda p: p == 'tp')).size())
        tn = sum_values(pred_labels.filter(payload_satisfies(lambda p: p == 'tn')).size())
        fp = sum_values(pred_labels.filter(payload_satisfies(lambda p: p == 'fp')).size())
        fn = sum_values(pred_labels.filter(payload_satisfies(lambda p: p == 'fn')).size())

        precision = tp / (tp + fp) if tp + fp > 0 else 0.
        recall = tp / (tp + fn)  if tp + fn > 0 else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0. else 0.

        return (precision, recall, f1, tp, tn, fp, fn)
    
    return precision_recall_f1(prediction_scores)

def commercials_query(captions, black_frames, whole_video, params={}):
    magic_numbers = {
        'RELIABLE_TEXT_DURATION': 5,
        'BLACKFRAME_COALESCE_EPSILON': 2,
        'CAPTIONS_COALESCE_EPSILON': 2,
        'COMMERCIAL_FOLD_EPSILON': 5,
        'MIN_COMMERCIAL_TIME': 10,
        'MAX_COMMERCIAL_TIME': 300,
        'LOWERCASE_COALESCE_EPSILON': 2,
        'MIN_LOWERTEXT': 0.5,
        'MIN_LOWERWINDOW': 15,
        'MAX_LOWERWINDOW_GAP': 60,
        'MIN_BLANKWINDOW': 30,
        'MAX_BLANKWINDOW': 270,
        'MAX_MERGE_GAP': 120,
        'MAX_MERGE_DURATION': 300,
    }
    
    magic_numbers.update(params)
    
    RELIABLE_TEXT_DURATION = magic_numbers['RELIABLE_TEXT_DURATION']
    BLACKFRAME_COALESCE_EPSILON = magic_numbers['BLACKFRAME_COALESCE_EPSILON']
    CAPTIONS_COALESCE_EPSILON = magic_numbers['CAPTIONS_COALESCE_EPSILON']
    COMMERCIAL_FOLD_EPSILON = magic_numbers['COMMERCIAL_FOLD_EPSILON']
    MIN_COMMERCIAL_TIME = magic_numbers['MIN_COMMERCIAL_TIME']
    MAX_COMMERCIAL_TIME = magic_numbers['MAX_COMMERCIAL_TIME']
    LOWERCASE_COALESCE_EPSILON = magic_numbers['LOWERCASE_COALESCE_EPSILON']
    MIN_LOWERTEXT = magic_numbers['MIN_LOWERTEXT']
    MIN_LOWERWINDOW = magic_numbers['MIN_LOWERWINDOW']
    MAX_LOWERWINDOW_GAP = magic_numbers['MAX_LOWERWINDOW_GAP']
    MIN_BLANKWINDOW = magic_numbers['MIN_BLANKWINDOW']
    MAX_BLANKWINDOW = magic_numbers['MAX_BLANKWINDOW']
    MAX_MERGE_GAP = magic_numbers['MAX_MERGE_GAP']
    MAX_MERGE_DURATION = magic_numbers['MAX_MERGE_DURATION']
    
    black_windows = frame_to_second_collection(black_frames.coalesce(
        ('t1', 't2'),
        Bounds3D.span,
        epsilon = BLACKFRAME_COALESCE_EPSILON
    ), cast_to_int = False)
    
    arrow_intervals = captions.filter(
        lambda intrvl: '>>' in intrvl['payload'] and '{' not in intrvl['payload']
    )
    arrow_announcer_intervals = captions.filter(
        lambda intrvl: '>> Announcer:' in intrvl['payload'] and '{' not in intrvl['payload']
    )
    arrow_having_intervals = captions.filter(
        lambda intrvl: '>> HAVING' in intrvl['payload'] and '{' not in intrvl['payload']
    )
    
    transcript_intervals = captions.filter(
        lambda intrvl: '{' not in intrvl['payload']
    ).coalesce(
        ('t1', 't2'),
        Bounds3D.span,
        epsilon = CAPTIONS_COALESCE_EPSILON
    )
    
    reliable_transcripts = transcript_intervals.filter_size(min_size = RELIABLE_TEXT_DURATION)
    arrow_intervals = arrow_intervals.minus(
        arrow_announcer_intervals
    ).minus(
        arrow_having_intervals
    ).filter_against(
        reliable_transcripts,
        predicate = overlaps()
    )
    
    all_blocks = whole_video.minus(black_windows)
    non_commercial_blocks = all_blocks.filter_against(
        arrow_intervals,
        predicate = overlaps()
    )
    
    commercial_blocks = whole_video.minus(non_commercial_blocks.union(black_windows))
    
    def fold_fn(stack, interval):
        if interval['t2'] - interval['t1'] > MAX_COMMERCIAL_TIME:
            interval = Interval(
                Bounds3D(interval['t1'], interval['t1'] + MAX_COMMERCIAL_TIME))
        if len(stack) == 0:
            stack.append(interval)
        else:
            last = stack.pop()
            if or_pred(overlaps(), after(max_dist=COMMERCIAL_FOLD_EPSILON))(interval, last):
                if last['bounds'].span(interval['bounds']).size() > MAX_COMMERCIAL_TIME:
                    stack.append(Interval(
                        Bounds3D(
                            last['t1'], 
                            last['t1'] + MAX_COMMERCIAL_TIME)))
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
    ).filter_size(min_size = MIN_COMMERCIAL_TIME)
    commercials_orig = commercials
    
    def is_lower_text(text):
        lower = [c for c in text if c.islower()]
        alpha = [c for c in text if c.isalpha()]
        if len(alpha) == 0:
            return False
        if 1. * len(lower) / len(alpha) > MIN_LOWERTEXT:
            return True
        else:
            return False
    
    lowercase_intervals = captions.filter(
        lambda intrvl: is_lower_text(intrvl['payload'])
    ).coalesce(
        ('t1', 't2'),
        Bounds3D.span,
        payload_merge_op = lambda p1, p2: p1 + ' ' + p2,
        epsilon = LOWERCASE_COALESCE_EPSILON
    ).filter_size(min_size = MIN_LOWERWINDOW)
    
    commercials = commercials.union(lowercase_intervals)
    
    blank_intervals = whole_video.minus(
        transcript_intervals
    ).filter_size(
        min_size=MIN_BLANKWINDOW, max_size=MAX_BLANKWINDOW
    ).minus(
        whole_video.map(
            lambda intrvl: Interval(Bounds3D(intrvl['t2'] - 60, intrvl['t2']))
        )
    ).filter_size(min_size=MIN_BLANKWINDOW)
    
    commercials = commercials.union(blank_intervals)
    
    commercials = commercials.coalesce(
        ('t1', 't2'),
        Bounds3D.span,
        epsilon = MAX_MERGE_GAP
    ).filter_size(
        max_size = MAX_COMMERCIAL_TIME
    ).union(
        commercials_orig
    ).union(
        lowercase_intervals
    ).union(
        blank_intervals
    ).coalesce(
        ('t1', 't2'),
        Bounds3D.span
    )
    
    return commercials