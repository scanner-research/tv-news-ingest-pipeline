import codecs
import json
import math
from multiprocessing import Pool
import os
import pickle
import re
import sys
import tempfile
import time

import cv2
import gentle
import numpy as np
import pysrt
import scipy.io.wavfile as wavf
from tqdm import tqdm

from util.consts import FILE_ALIGNMENT_STATS, FILE_CAPTIONS
from util.utils import get_base_name, save_json

#----------Help functions for fid, time, second transfer----------
def time2second(time):
    return time[0]*3600 + time[1]*60 + time[2] + time[3] / 1000.0


def second2time(second, sep=','):
    h = int(second) // 3600
    m = int(second % 3600) // 60
    s = int(second) % 60
    ms = int((second - int(second)) * 1000)
    return '{:02d}:{:02d}:{:02d}{:s}{:03d}'.format(h, m, s, sep, ms)


#---------Forced transcript-audio alignment using gentle----------
class TranscriptAligner():
    def __init__(self, win_size=300, seg_length=60, max_misalign=10,
                 num_thread=1, estimate=True, missing_thresh=0.2,
                 transcript_path=None, media_path=None, align_dir=None):
        """
        @win_size: chunk size for estimating maximum mis-alignment
        @seg_length: chunk size for performing gentle alignment
        @max_misalign: maximum mis-alignment applied at the two ends of seg_length
        @num_thread: number of threads used in gentle
        @estimate: if True, run maximum mis-alignment estimate on each chunk of win_size if the first run failed
        @missing_thresh: the threshold for acceptable missing word rate
        @transcript_path: path to original transcript
        @media_path: path to video/audio
        @align_dir: path to save aligned transcript

        """

        self.win_size = win_size
        self.seg_length = seg_length
        self.text_shift = max_misalign
        self.num_thread = num_thread
        self.estimate = estimate
        self.missing_thresh = missing_thresh
        self.transcript_path = transcript_path
        self.media_path = media_path
        self.align_dir = align_dir

        self.audio_shift = 1
        self.clip_length = 15
        self.seg_idx = 0
        self.punctuation_all = ['>>', ',', ':', '[.]', '[?]']
        self.num_words = 0

        if self.media_path is not None:
            _, ext = os.path.splitext(self.media_path)
            self.video_name = os.path.basename(self.media_path).split('.')[0]
            if ext == '.mp4':
                cap = cv2.VideoCapture(self.media_path)
                self.video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                self.video_length = int(self.video_frames // self.fps)
                self.num_seg = int(self.video_length // self.seg_length)
                self.num_window = int(self.video_length // self.win_size)
            else:
                raise Exception("File type '{}' not supported.".format(ext))

        if align_dir is not None:
            os.makedirs(align_dir, exist_ok=True)

        self.pbar = None


    def load_transcript(self, transcript_path):
        """
        Load transcript from .srt file.

        """

        # Check file exist
        if not os.path.exists(transcript_path):
            raise Exception("Transcript file does not exist")

        # Check encoded in uft-8
        try:
            file = codecs.open(transcript_path, encoding='utf-8', errors='strict')
            for line in file:
                pass
        except UnicodeDecodeError:
            raise Exception("Transcript not encoded in utf-8")

        transcript = []
        subs = pysrt.open(transcript_path)
        text_length = 0
        num_words = 0
        for sub in subs:
            transcript.append((sub.text, time2second(tuple(sub.start)[:4]), time2second(tuple(sub.end)[:4])))
            text_length += transcript[-1][2] - transcript[-1][1]
            for w in sub.text.replace('.', ' ').replace('-', ' ').split():
                num_words += 1 if w.islower() or w.isupper() else 0

        self.transcript = transcript
        self.num_words = num_words


    def extract_transcript(self, start, end, offset_to_time=False):
        """
        Extract transcript between [start, end) into a string, with additional
        offset to timestamp/puncuation.

        """

        text_total = ''
        if offset_to_time:
            offset2time = {}
        else:
            offset2punc = []
        for (text, ss, ee) in self.transcript:
            if ss >= end:
                break
            if ss >= start:
                offset = len(text_total)
                if offset_to_time:
                    words = text.replace('.', ' ').replace('-', ' ').split(' ')
                    step = (ee - ss) / len(words)
                    for i, w in enumerate(words):
                        offset2time[offset] = ss+i*step
                        offset += len(w) + 1
                else:
                    for p in self.punctuation_all:
                        pp = p.replace('[', '').replace(']', '')
                        for match in re.finditer(p, text):
                            offset2punc.append((match.start()+offset, pp))
                text_total += text + ' '
        if offset_to_time:
            return text_total, offset2time
        else:
            offset2punc.sort()
            return text_total, offset2punc


    def extract_transcript_segment(self, seg_idx, large_shift=0):
        """
        Extract transcript given specific segment.

        """

        start = seg_idx * self.seg_length - self.text_shift - large_shift
        end = (seg_idx + 1) * self.seg_length + self.text_shift - large_shift
        return self.extract_transcript(start, end, offset_to_time=False)


    def extract_transcript_all(self, estimate=False):
        """
        Extract transcript from all segments.

        """

        self.text_seg_list = []
        self.punc_seg_list = []
        for seg_idx in range(self.num_seg):
            shift = self.shift_seg_list[seg_idx] if estimate else 0
            transcript_seg, punctuation_seg = self.extract_transcript_segment(seg_idx, shift)
            self.text_seg_list.append(transcript_seg)
            self.punc_seg_list.append(punctuation_seg)


    def extract_audio(self, start, end):
        """
        Extract audio between [start, end] into a local .aac file.

        """

        duration = end - start
        cmd = 'ffmpeg -loglevel error -i ' + self.media_path + ' -vn -acodec copy '
        cmd += '-ss {:d} -t {:d} '.format(start, duration)
        audio_path = tempfile.NamedTemporaryFile(suffix='.aac').name
        cmd += audio_path
        os.system(cmd)
        return audio_path


    def extract_audio_segment(self, seg_idx):
        """
        Extract audio given specific segment.
        """

        start = seg_idx * self.seg_length
        start = start - self.audio_shift if seg_idx > 0 else start
        duration = self.seg_length
        duration += self.audio_shift * 2 if seg_idx > 0 else self.audio_shift
        return self.extract_audio(start, start + duration)


    def extract_audio_all(self):
        """
        Extract audio from all segments parallely.

        """

        pool = Pool(self.num_thread)
        self.audio_seg_list = pool.map(self.extract_audio_segment,
                                       [i for i in range(self.num_seg)])


    def gentle_solve(self, audio_path, transcript):
        """
        Gentle wrapper to solve the forced alignment given audio file and
        text string.

        """

        args = {
            'log': 'INFO',
            'nthreads': 1,
            'conservative': True,
            'disfluency': True,
        }
        disfluencies = set(['uh', 'um'])
        resources = gentle.Resources()
        with gentle.resampled(audio_path) as wavfile:
            aligner = gentle.ForcedAligner(resources, transcript,
                                           nthreads=args['nthreads'],
                                           disfluency=args['disfluency'],
                                           conservative=args['conservative'],
                                           disfluencies=disfluencies)
            result = aligner.transcribe(wavfile)

        return [word.as_dict(without="phones") for word in result.words]


    def align_segment_thread(self, seg_idx):
        """
        Wrapper function for multiprocessing.

        """
        return self.align_segment(seg_idx, self.audio_seg_list[seg_idx],
                                  self.text_seg_list[seg_idx],
                                  self.punc_seg_list[seg_idx])


    def align_segment(self, seg_idx, audio_path, transcript, punctuation):
        """
        Call gentle and post-process aligned results.

        """

        aligned_seg = self.gentle_solve(audio_path, transcript)

        # Insert punctuation
        start_idx = 0
        for offset, p in punctuation:
            for word_idx, word in enumerate(aligned_seg[start_idx:]):
                if word['case'] != 'not-found-in-transcript':
                    if p == '>>' and (offset == word['startOffset'] - 3 or offset == word['startOffset'] - 4):
                        word['word'] = '>> ' + word['word']
                        start_idx += word_idx
                        break
                    if p != '>>' and offset == word['endOffset']:
                        word['word'] = word['word'] + p
                        start_idx += word_idx
                        break

        # post-process
        align_word_list = []
        seg_start = seg_idx * self.seg_length
        seg_start = seg_start - self.audio_shift if seg_idx > 0 else seg_start
        seg_shift = self.audio_shift if seg_idx > 0 else 0

        enter_alignment = False
        word_missing = []
        num_word_aligned = 0
        for word_idx, word in enumerate(aligned_seg):
            if word['case'] == 'not-found-in-transcript':
                # align_word_list.append(('[Unknown]', (word['start'] + seg_start, word['end'] + seg_start)))
                pass
            elif word['case'] == 'not-found-in-audio':
                if enter_alignment:
                    word_missing.append(word['word'])
            else:
                assert(word['case'] == 'success')
                if word['start'] > self.seg_length + seg_shift:
                    break
                elif word['start'] >= seg_shift:
                    num_word_aligned += 1
                    enter_alignment = True
                    cur_start = word['start'] + seg_start
                    cur_end = word['end'] + seg_start

                    # make sure the prev_end <= cur_start
                    if len(align_word_list) > 0:
                        prev_end = align_word_list[-1][1][1]
                        if prev_end > cur_start and prev_end < cur_end:
                            cur_start = prev_end

                    # mis-aligned word handling
                    if len(word_missing) <= 2:
                        num_word_aligned += len(word_missing)
                    if len(word_missing) > 0:
                        step = (cur_start - prev_end) / len(word_missing)
                        for i, w in enumerate(word_missing):
                            align_word_list.append(('{'+w+'}', (prev_end+i*step, prev_end+(i+1)*step)))
                        word_missing = []
                    align_word_list.append((word['word'], (cur_start, cur_end)))

        return {'align_word_list': align_word_list, 'num_word_aligned': num_word_aligned}


    def estimate_shift_clip(self, audio_path, audio_start, transcript, offset2time):
        """
        Given an audio clip, call gentle and then estimate a rough misalignment.

        """

        aligned_clip = self.gentle_solve(audio_path, transcript)

        align_word_list = []
        shift_list = []
        for word_idx, word in enumerate(aligned_clip):
            if word['case'] == 'success':
                if word['startOffset'] in offset2time:
                    shift = word['start'] + audio_start - offset2time[word['startOffset']]
                    if np.abs(shift) <= self.win_size:
                        shift_list.append(shift)
#                 else:
#                     shift = 0
#                 align_word_list.append((word['word'], word['start'] + audio_start, shift))
        l = len(shift_list)
        if l < 4:
            return None
        else:
            return np.average(shift_list[l//4 : l*3//4])


    def estimate_shift_window(self, win_idx):
        """
        Estimate rough misalignment given a specific window.

        """

        transcript, offset2time = self.extract_transcript((win_idx-1)*self.win_size, (win_idx+2)*self.win_size, offset_to_time=True)
        shift_list = []
        for audio_shift in range(self.seg_length//2, self.win_size, self.seg_length):
            audio_start = win_idx * self.win_size + audio_shift
            audio_path = self.extract_audio(audio_start, audio_start + self.clip_length)
            shift = self.estimate_shift_clip(audio_path, audio_start, transcript, offset2time)
            if not shift is None:
                shift_list.append(shift)
        if len(shift_list) == 0:
            return 0
        else:
            shift_list.sort()
            return np.median(shift_list)


    def estimate_shift_all(self):
        """
        Estimate rough mis-alignment for all windows.

        """
        with Pool(self.num_thread) as workers, tqdm(
            total=self.num_window,
            desc='Estimating shift window',
            unit='window'
        ) as pbar:
            results = [workers.apply_async(
                self.estimate_shift_window,
                args=(i,),
                callback=lambda x: pbar.update()
            ) for i in range(self.num_window)]

            workers.close()
            workers.join()

        shift_window_list = [r.get() for r in results]
        shift_seg_list = []
        for shift in shift_window_list:
            shift_seg_list += [shift] * (self.win_size // self.seg_length)
        if shift_seg_list:
            shift_seg_list += [shift_seg_list[-1]] * (self.num_seg - len(shift_seg_list))

        return shift_seg_list


    def prepare_input(self):
        """
        Prepare transcript and audio segments for aligning.

        """

        self.load_transcript(self.transcript_path)
        self.extract_audio_all()


    def run(self):
        """
        Entrypoint for solving transcript-audio alignment.

        """

        self.prepare_input()
        self.extract_transcript_all()

        with Pool(self.num_thread) as workers, tqdm(total=self.num_seg,
            desc='Aligning captions without estimating shift',
            unit='segment'
        ) as pbar:
            results = [
                workers.apply_async(
                    self.align_segment_thread,
                    args=(i,),
                    callback=lambda x: pbar.update()
                ) for i in range(self.num_seg)
            ]

            workers.close()
            workers.join()

        result_all = [r.get() for r in results]

        align_word_list = []
        num_word_aligned = 0
        for seg_idx, seg in enumerate(result_all):
            align_word_list += [word for word in seg['align_word_list']]
            num_word_aligned += seg['num_word_aligned']

        missing_rate = 1 - 1. * num_word_aligned / self.num_words

        if self.estimate and missing_rate > self.missing_thresh:
            self.shift_seg_list = self.estimate_shift_all()
            self.extract_transcript_all(estimate=True)

            # Second: run with estimating the shift
            with Pool(self.num_thread) as workers, tqdm(total=self.num_seg,
                desc='Aligning captions with estimating shift',
                unit='segment'
            ) as pbar:
                results = [
                    workers.apply_async(
                        self.align_segment_thread,
                        args=(i,),
                        callback=lambda x: pbar.update()
                    ) for i in range(self.num_seg)
                ]

                workers.close()
                workers.join()

            result_all = [r.get() for r in results]

            align_word_list = []
            num_word_aligned = 0
            for seg_idx, seg in enumerate(result_all):
                align_word_list += [word for word in seg['align_word_list']]
                num_word_aligned += seg['num_word_aligned']

        if not self.align_dir is None:
            output_path = os.path.join(self.align_dir, FILE_CAPTIONS)
            self.dump_aligned_transcript_byword(align_word_list, output_path)

        return {'word_missing': 1 - 1. * num_word_aligned / self.num_words}


    @staticmethod
    def dump_aligned_transcript(align_word_list, path):
        SRT_INTERVAL = 1
        outfile = open(path, 'w')
        start, end = None, None
        srt_idx = 1
        for idx, word in enumerate(align_word_list):
            if start is None:
                start, end = word[1]
                text = word[0] + ' '
                continue
            if word[1][0] > start + SRT_INTERVAL:
                line = str(srt_idx) + '\n'
                line += '{:s} --> {:s}\n'.format(second2time(start), second2time(end))
                line += text + '\n\n'
                outfile.write(line)
                start, end = word[1]
                text = word[0] + ' '
                srt_idx += 1
            else:
                text += word[0] + ' '
                end = word[1][1]
        line = str(srt_idx) + '\n'
        line += '{:s} --> {:s}\n'.format(second2time(start), second2time(end))
        line += text + '\n\n'
        outfile.write(line)
        outfile.close()


    @staticmethod
    def dump_aligned_transcript_byword(align_word_list, path):
        outfile = open(path, 'w')
        srt_idx = 1
        for idx, word in enumerate(align_word_list):
            start, end = word[1]
            line = str(srt_idx) + '\n'
            line += '{:s} --> {:s}\n'.format(second2time(start), second2time(end))
            line += word[0] + '\n\n'
            outfile.write(line)
            srt_idx += 1
        outfile.close()


def main(video_in_path, transcript_in_path, out_path, force=False):
    if video_in_path.endswith('.mp4') and transcript_in_path.endswith('.srt'):
        video_paths = [video_in_path]
        transcript_paths = [transcript_in_path]
    else:
        video_paths = [
            l.strip() for l in open(video_in_path, 'r') if l.strip()
        ]
        transcript_paths = [
            l.strip() for l in open(transcript_in_path, 'r') if l.strip()
        ]

    assert len(video_paths) == len(transcript_paths), \
           'There was a mismatch between the number of videos and transcripts.'

    video_paths = sorted(video_paths, key=get_base_name)
    transcript_paths = sorted(transcript_paths, key=get_base_name)

    video_names = [get_base_name(p) for p in video_paths]

    assert all(get_base_name(t) == video_names[i]
               for i, t in enumerate(transcript_paths)), \
           'There was a mismatch between videos and transcript names.'

    out_paths = [os.path.join(out_path, name) for name in video_names]

    for p in out_paths:
        os.makedirs(p, exist_ok=True)

    num_threads = os.cpu_count() if os.cpu_count() else 1
    for i in range(len(video_names)):
        if not force and os.path.exists(os.path.join(out_paths[i],
                                                     FILE_CAPTIONS)):
            print("Skipping aligning captions for video '{}': captions file " \
                  "already exists!".format(
                  video_names[i]))
            continue

        print("Aligning captions for video '{}'".format(video_names[i]))

        aligner = TranscriptAligner(win_size=300, seg_length=60,
            max_misalign=10, num_thread=num_threads, estimate=True, missing_thresh=0.2,
            media_path=video_paths[i],
            transcript_path=transcript_paths[i],
            align_dir=out_paths[i])

        stats = aligner.run()
        save_json(stats, os.path.join(out_paths[i], FILE_ALIGNMENT_STATS))
