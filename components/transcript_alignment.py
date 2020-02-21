import numpy as np
import pickle
import pysrt
import codecs
import math
import sys
import os
import cv2
import logging
import time
import re
import tempfile
import scipy.io.wavfile as wavf
import multiprocessing
import json
import traceback
import gentle

#----------Help functions for fid, time, second transfer----------
def fid2second(fid, fps):
    second = 1. * fid / fps
    return second


def time2second(time):
    return time[0]*3600 + time[1]*60 + time[2] + time[3] / 1000.0


def second2time(second, sep=','):
    h, m, s, ms = int(second) // 3600, int(second % 3600) // 60, int(second) % 60, int((second - int(second)) * 1000)
    return '{:02d}:{:02d}:{:02d}{:s}{:03d}'.format(h, m, s, sep, ms)


#---------Forced transcript-audio alignment using gentle----------
class TranscriptAligner():
    def __init__(self, win_size=300, seg_length=60, max_misalign=10, num_thread=1, estimate=True, missing_thresh=0.2,
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
            elif ext == '.wav':
                raise Exception("Not implemented error")
        if align_dir is not None:
            os.makedirs(align_dir, exist_ok=True)
    
    
    def load_transcript(self, transcript_path):
        """
        Load transcript from *.srt file
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
        print('Num of words in transcript:',  num_words)

        self.transcript = transcript
        self.num_words = num_words
        
        
    def extract_transcript(self, start, end, offset_to_time=False):
        """
        extract transcript between [start, end) into a string, with additional offset to timestamp/puncuation
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
        extract transcript given specific segment
        """
        start = seg_idx * self.seg_length - self.text_shift - large_shift
        end = (seg_idx + 1) * self.seg_length + self.text_shift - large_shift
        return self.extract_transcript(start, end, offset_to_time=False)
    
    
    def extract_transcript_all(self, estimate=False):
        """
        extract transcript from all segments
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
        extract audio between [start, end] into a local .aac file
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
        extract audio given specific segment
        """
        start = seg_idx * self.seg_length
        start = start - self.audio_shift if seg_idx > 0 else start
        duration = self.seg_length 
        duration += self.audio_shift * 2 if seg_idx > 0 else self.audio_shift
        return self.extract_audio(start, start + duration)
    
    
    def extract_audio_all(self):
        """
        extract audio from all segments parallely
        """
        pool = multiprocessing.Pool(self.num_thread)
        self.audio_seg_list = pool.map(self.extract_audio_segment, [i for i in range(self.num_seg)])
    
    
    def gentle_solve(self, audio_path, transcript):
        """
        gentle wrapper to solve the forced alignment given audio file and text string 
        """
        args = {'log': 'INFO',
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
        function wrapped for multiprocessing
        """
        return self.align_segment(seg_idx, self.audio_seg_list[seg_idx], self.text_seg_list[seg_idx], self.punc_seg_list[seg_idx])
    
    
    def align_segment(self, seg_idx, audio_path, transcript, punctuation):
        """
        call gentle and post-process aligned results
        """
        print('Starting gentle to align %dth segment' % seg_idx)
        aligned_seg = self.gentle_solve(audio_path, transcript)
        print('Finished gentle to align %dth segment' % seg_idx)
        
        # insert punctuation
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
        Given an audio clip, call gentle and then estimate a rough mis-alignment
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
#         print(align_word_list)
        l = len(shift_list)
        if l < 4:
            return None
        else:
            return np.average(shift_list[l//4 : l*3//4])
    
    
    def estimate_shift_window(self, win_idx):
        """
        Estimate rough mis-alignment given a specific window
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
        Estimate rough mis-alignment for all windows 
        """
        pool = multiprocessing.Pool(self.num_thread)
        shift_window_list = pool.map(self.estimate_shift_window, [i for i in range(self.num_window)])
        shift_seg_list = []
        for shift in shift_window_list:
            shift_seg_list += [shift] * (self.win_size // self.seg_length)
        shift_seg_list += [shift_seg_list[-1]] * (self.num_seg - len(shift_seg_list))
        return shift_seg_list
    
    
    def prepare_input(self):
        """
        Prepare transcript and audio segments for aligning
        """
        self.load_transcript(self.transcript_path)
        
        self.extract_audio_all()
        print("Extracting audio done")
       
        print('Total number of segments: %d' % self.num_seg)
    
    
    def run(self):
        """
        Entrance for solving transcript-audio alignment
        """
        
        self.prepare_input()
        
        self.extract_transcript_all()
        
        # First: run without estimating the shift
        print("Starting alignment without estimating shift...")
        print('')
        pool = multiprocessing.Pool(self.num_thread)
        result_all = pool.map(self.align_segment_thread, [i for i in range(self.num_seg)])
           
        align_word_list = []
        num_word_aligned = 0
        for seg_idx, seg in enumerate(result_all):
            align_word_list += [word for word in seg['align_word_list']]
            num_word_aligned += seg['num_word_aligned']
        
        missing_rate = 1 - 1. * num_word_aligned / self.num_words
        print("Missing word after first run: %.2f " % missing_rate)
        
        if self.estimate and missing_rate > self.missing_thresh:
            print("Starting alignment with estimating shift...")
            print('')
            self.shift_seg_list = self.estimate_shift_all()
            print("Estimating shift done")
            self.extract_transcript_all(estimate=True)

            # Second: run with estimating the shift
            pool = multiprocessing.Pool(self.num_thread)
            result_all = pool.map(self.align_segment_thread, [i for i in range(self.num_seg)])

            align_word_list = []
            num_word_aligned = 0
            for seg_idx, seg in enumerate(result_all):
                align_word_list += [word for word in seg['align_word_list']]
                num_word_aligned += seg['num_word_aligned']
                
        print('num_word_total: ', self.num_words)
        print('num_word_aligned: ', num_word_aligned)
        print('word_missing by total words: ', 1 - 1. * num_word_aligned / self.num_words)
        print('word_missing by total aligned: ', 1 - 1. * num_word_aligned / len(align_word_list))
        
        if not self.align_dir is None:
            output_path = os.path.join(self.align_dir, self.video_name + '.word.srt')
            self.dump_aligned_transcript_byword(align_word_list, output_path)
#             output_path = os.path.join(self.align_dir, self.video_name + '.align.srt')
#             self.dump_aligned_transcript(align_word_list, output_path)
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

        
if __name__ == "__main__":
    output_dir='/app/data/subs/aligned_kdd_first'
    transcript_dir = '/app/data/subs/subs_kdd/'
    video_dir = '/app/data/videos/'
    google_storage_dir = 'gs://esper/tvnews/videos/'
    result_pkl = '/app/result/aligned_kdd_first.pkl'
    
    video_list = open('/app/data/video_list_kdd2.txt', 'r').read().split('\n')[:-1]
    
    if os.path.exists(result_pkl):
        res_stats = pickle.load(open(result_pkl, 'rb'))
    else:
        res_stats = {}
    for video_name in video_list:
        print(video_name)
        if video_name in res_stats:
            continue
            
        video_path = os.path.join(video_dir, video_name + '.mp4')
        transcript_path=os.path.join(transcript_dir, video_name)
        
        # Download video to local
        if not os.path.exists(video_path):
            gs_path = os.path.join(google_storage_dir, video_name + '.mp4')
            cmd = 'gsutil cp ' + gs_path + ' ' + video_dir
            print(cmd)
            os.system(cmd)
            print('Downloading video done')
        else:
            print('Video exists!')
        
        start_time = time.time()
        
        aligner = TranscriptAligner(win_size=300, seg_length=60, max_misalign=10, num_thread=64, estimate=True, missing_thresh=0.2,
                        media_path=video_path,
                        transcript_path=transcript_path,
                        align_dir=output_dir)
        
        # First attempt 
        res = aligner.run()
        res_stats[video_name] = res
        print('Alignment finished in %f s' % (time.time() - start_time))
        
        # Delete video
        cmd = 'rm ' + video_path
        os.system(cmd)
        print('Delete video done')

        pickle.dump(res_stats, open(result_pkl, 'wb'))