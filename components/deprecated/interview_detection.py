from rekall import Interval, IntervalSet, IntervalSetMapping, Bounds3D
from rekall.predicates import *
from rekall.stdlib import ingest
from rekall.stdlib.merge_ops import *
from vgrid import VGridSpec, VideoMetadata, VideoBlockFormat, FlatFormat
from vgrid import SpatialType_Bbox, SpatialType_Caption, Metadata_Generic
from vgrid_jupyter import VGridWidget
import math

import json
import os
import sys
from collections import OrderedDict, defaultdict
from typing import Optional, NamedTuple, List
from rs_intervalset import MmapIntervalListMapping, MmapIntervalSetMapping


HOST_FILE = '../data/hosts.txt'
GUEST_FILE = '../data/guests.txt'

VIDEOS_FILE = '../data/videos.json'
NUM_FACES_FILE = '../data/num_faces.ilist.bin'
COMMERCIALS_FILE = '../data/commercials.iset.bin'

PERSON_ILIST_DIRS = [
    '../data/people',   # AWS
    '../data/people2'   # ours
]

class Video(NamedTuple):
    id: int
    name: str
    show: str
    channel: str
    fps: float
    duration: float
    num_frames: int
    width: int
    height: int

class Person(NamedTuple):
    name: str
    channel: str
    ilist: MmapIntervalListMapping
        
def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)
        
def load_videos():
    def get_video_name(v):
        return os.path.splitext(v.split('/')[-1])[0]

    videos = OrderedDict()
    for v in sorted(load_json(VIDEOS_FILE)):
        (
            id,
            name,
            show,
            channel,
            num_frames,
            fps,
            width,
            height
        ) = v
        assert isinstance(id, int)
        assert isinstance(name, str)
        assert isinstance(show, str)
        assert isinstance(channel, str)
        assert isinstance(num_frames, int)
        assert isinstance(fps, float)
        assert isinstance(width, int)
        assert isinstance(height, int)

        video_name = get_video_name(name)
        videos[id] = Video(
            id=id, name=video_name, show=show, channel=channel,
            duration=num_frames / fps, fps=fps, num_frames=num_frames, width=width, height=height)
    return videos
        
def load_hosts():
    people = []
    with open(HOST_FILE) as fp:
        for l in fp:
            l = l.strip()
            if not l:
                continue
            channel, name = l.split(',', 1)

            for dir in PERSON_ILIST_DIRS:
                ilist_path = os.path.join(
                    dir, '{}.ilist.bin'.format(name))
                if os.path.isfile(ilist_path):
                    break
            else:
                print('No interval list:', name)
                continue

            ilist = MmapIntervalListMapping(ilist_path, 1)
            people.append(Person(name, channel, ilist))
    return people


def load_guests():
    people = []
    with open(GUEST_FILE) as fp:
        for l in fp:
            l = l.strip()
            if not l:
                continue
            name = l

            for dir in PERSON_ILIST_DIRS:
                ilist_path = os.path.join(
                    dir, '{}.ilist.bin'.format(name))
                if os.path.isfile(ilist_path):
                    break
            else:
                print('No interval list:', name)
                continue

            ilist = MmapIntervalListMapping(ilist_path, 1)
            people.append(Person(name, None, ilist))
    return people


def load_num_faces():
    return MmapIntervalListMapping(NUM_FACES_FILE, 1)


def rs_to_rekall(rs_ilist, video_ids=None, with_payload=True):
    rekall_ism = {}
    if video_ids is None:
        video_ids = rs_ilist.get_ids()
    for video_id in video_ids:
        if with_payload:
            interval_list = rs_ilist.get_intervals_with_payload(video_id, True)
            rekall_ism[video_id] = IntervalSet([
                Interval(
                    Bounds3D(
                    i[0] / 1000., i[1] / 1000., 0, 0, 0, 0),
                    i[2]
                )
                for i in interval_list])
        else:
            interval_list = rs_ilist.get_intervals(video_id, True)
            rekall_ism[video_id] = IntervalSet([
                Interval(
                    Bounds3D(
                    i[0] / 1000., i[1] / 1000., 0, 0, 0, 0)
                )
                for i in interval_list])
        
    return IntervalSetMapping(rekall_ism)


def ism_to_json(ism):
    return {video_id: ism[video_id].to_json(lambda x: None) for video_id in ism}


def interviews_query_kdd(guest, host, commercial, num_faces):
    # Magic numbers
    GUEST_BEFORE_OR_AFTER_DISTANCE = 60
    INTERVIEW_CANDIDATE_COALESCE_EPSILON = 120
    INTERVIEW_MIN_SIZE = 240    
        
    # Remove single frame intervals 
#     person_intrvlcol = person_intrvlcol.filter_length(min_length=1)
#     host_intrvlcol = host_intrvlcol.filter_length(min_length=1)
    
    
    # Remove isolated small intervals
#     host_intrvlcol = remove_isolated_interval(host_intrvlcol)
#     person_intrvlcol = remove_isolated_interval(person_intrvlcol)
    
    
    # Split long segments into connected pieces
#     host_intrvlcol = split_intrvlcol(host_intrvlcol, seg_length=30)


    # This temporal predicate defines A overlaps with B, or A before by less than 60 seconds,
    #   or A after B by less than 60 seconds
    interviews = guest.join(
        host,
        predicate = or_pred(
            before(max_dist = GUEST_BEFORE_OR_AFTER_DISTANCE),
            after(max_dist = GUEST_BEFORE_OR_AFTER_DISTANCE),
            overlaps()
        ),
        merge_op = lambda i1, i2: Interval(i1['bounds'].span(i2['bounds'])),
        window = GUEST_BEFORE_OR_AFTER_DISTANCE
    ).coalesce(
        ('t1', 't2'),
        Bounds3D.span,
        epsilon = INTERVIEW_CANDIDATE_COALESCE_EPSILON
    ).minus(commercial
    ).filter_size(min_size = INTERVIEW_MIN_SIZE)
    
    
    # Remove interview segments which the total person time proportion is below threshold
    def filter_guest_time(i):
        # Thresh for Trump 0.4, 0.4, 0.5 
        guest_time, small_guest_time, large_guest_time = 0, 0, 0
        for encode, duration in i.payload:
            height = ((encode & 0b11111100) >> 2) / 100.
            guest_time += duration
            small_guest_time += duration if height < 0.3 else 0
            large_guest_time += duration if height > 0.3 else 0
        seg_length = i.size()
        return guest_time / seg_length > 0.35 and small_guest_time / guest_time < 0.7
    
    interviews_guest_time = interviews.join(
        guest,
        predicate=overlaps(),
        merge_op = lambda i1, i2: Interval(
            Bounds3D(**i1['bounds'].data), 
            [(i2['payload'], i2.size())]),
        window = 0,
        ).coalesce(
            ('t1', 't2'),
            Bounds3D.span,
            payload_merge_op=payload_plus)
    interviews = interviews_guest_time.filter(filter_guest_time)
    
    
    # Remove interview segments which the total host time proportion is below threshold
    # and there is only one host showing a lot
#     def filter_host_time(i):
#         host_time = {}
#         for faceID_id, duration in i.payload:
#             if not faceID_id in host_time:
#                 host_time[faceID_id] = 0
#             host_time[faceID_id] += duration
#         host_time_sort = sorted(host_time.values())
#         sum_host_time = sum(host_time_sort)
#         if sum_host_time / (i.end - i.start) < 0.1:
#             return False
#         if len(host_time_sort) > 1 and host_time_sort[-2] > 0.2:
#             return False
#         return True
    
    
    # Remove interview if the person is alone for too long
#     def filter_person_alone(i):
#         for duration in i.payload:
#             if duration / (i.end - i.start) > 0.5:
#                 return False
#         return True

#     interviews_person_alone = interviews.join(interviews.minus(host_intrvlcol),
#                              predicate=overlaps(),
#                              merge_op=lambda i1, i2: [(i1.start, i1.end, [i2.end - i2.start])] ) \
#                              .coalesce(payload_plus)
#     interviews = interviews_person_alone.filter(filter_person_alone)


    # Remove interview segments where many people showing a lot
    def filter_num_faces(i):
        multi_person_duration = sum([duration for nface, duration in i['payload'] if nface > 2])
        return multi_person_duration / i.size() < 0.05
    
    interviews_num_faces = interviews.join(
        num_faces,
        predicate = overlaps(),
        merge_op = lambda i1, i2: Interval(
            Bounds3D(**i1['bounds'].data), 
            [(i2['payload'], i2.size())]),
        window = 0,
        ).coalesce(
            ('t1', 't2'),
            Bounds3D.span,
            payload_merge_op=payload_plus)
    interviews = interviews_num_faces.filter(filter_num_faces)

    
    guest_all = interviews.join(
        guest,
        predicate = overlaps(),
        merge_op = lambda i1, i2: Interval(i1['bounds'].intersect_time_span_space(i2['bounds'])),
        window = 0
    )
    
    guest_with_host = guest_all.join(
        host,
        predicate = overlaps(),
        merge_op = lambda i1, i2: Interval(i1['bounds'].intersect_time_span_space(i2['bounds'])),
        window = 0
    )
    
    guest_only = guest_all.minus(guest_with_host)
    
    host_only = interviews.join(
        host,
        predicate = overlaps(),
        merge_op = lambda i1, i2: Interval(i1['bounds'].intersect_time_span_space(i2['bounds'])),
        window = 0
    ).minus(guest_with_host)
    
#     guest_all = interviews.overlaps(guest)

#     guest_with_host = guest_all.overlaps(host)
    
#     guest_only = guest_all.minus(guest_with_host)
    
#     host_only = interviews.overlaps(host).minus(guest_with_host)
    
    return interviews, guest_only, host_only, guest_with_host


if __name__ == '__main__':
    
    assert len(sys.argv) == 2, 'Missing input argument' 
    
    hosts = load_hosts()
    guests = load_guests()
    videos = load_videos()
    num_faces = load_num_faces()
    commercials = MmapIntervalSetMapping(COMMERCIALS_FILE)
    
    guest = [h for h in guests if h.name == sys.argv[1].replace('-', ' ')][0]
    print("Detecting interview for %s..." % guest.name)
    
    result = {guest.name: []}
    for host in hosts:
        print(host.name)
        host_ids = set(host.ilist.get_ids())
        guest_ids = set(guest.ilist.get_ids())
        video_ids = host_ids & guest_ids
        print(len(video_ids))

        host_ism = rs_to_rekall(host.ilist, video_ids)
        guest_ism = rs_to_rekall(guest.ilist, video_ids)
        num_faces_ism = rs_to_rekall(num_faces, video_ids)
        commercial_ism = rs_to_rekall(commercials, video_ids, with_payload=False)

        interviews, guest_only, host_only, guest_with_host = interviews_query_kdd(
            guest_ism, host_ism, commercial_ism, num_faces_ism)

        result[guest.name].append({'host': host.name,
                                   'interviews': ism_to_json(interviews),
                                   'guest_only': ism_to_json(guest_only),
                                   'host_only': ism_to_json(host_only),
                                   'guest_with_host': ism_to_json(guest_with_host)})

    json.dump(result, open('../result/interviews_{}.json'.format(guest.name), 'w'))