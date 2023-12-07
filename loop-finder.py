import os
import sys
import argparse
import numpy as np
import cv2
import imagehash
from PIL import Image
from datetime import timedelta, datetime
from moviepy.editor import VideoFileClip
from multiprocessing import Pool
from tqdm import tqdm
import magic

# Globals with placholders; will be replaced by command line parameters
full_video = None

MIN_LENGTH = 0.5
MAX_LENGTH = 10

THRESHOLD = 0.85
EVAL = "quality"
HASH_FUNC = imagehash.dhash

FPS = 100
HASH_SIZE = 64
BANDS = 32

def format_timedelta(td):
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00").replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")

def get_saving_frames_durations(cap, saving_fps):
    s = []
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def is_valid(start_str, end_str):
    start_obj = datetime.strptime(start_str, '%H-%M-%S.%f')
    end_obj   = datetime.strptime(end_str, '%H-%M-%S.%f')
    td = end_obj - start_obj
    return td >= timedelta(seconds=MIN_LENGTH) and td <= timedelta(seconds=MAX_LENGTH)

def bin_sum(tup):
    return (tup[0],tup[1],sum(tup[2]))

def calculate_signatures(split):
    saving_frames_per_second = min(full_video.fps, FPS)
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    
    i = 0
    signatures = dict()
    fd_to_idx = dict()
    frame_list = list()
    for current_duration in np.arange(split[0],split[1], step):
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration))
        frame = full_video.get_frame(current_duration)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        f = frame.convert("L").resize((HASH_SIZE+1,HASH_SIZE),Image.LANCZOS)
        
        hash_val = HASH_FUNC(f,HASH_SIZE)
        signature = hash_val.hash.flatten()
        
        signatures[frame_duration_formatted] = np.packbits(signature)
        fd_to_idx[frame_duration_formatted] = i
        frame_list.append(frame_duration_formatted)
    return signatures, fd_to_idx, frame_list

def find_similarity(signatures, fd_to_idx, frame_list):
    rows = int(HASH_SIZE**2/BANDS)
    hash_buckets_list = [dict() for _ in range(BANDS)]

    for fh, signature in signatures.items():
        for i in range(BANDS):
            signature_band = signature[i*rows:(i+1)*rows]
            signature_band_bytes = signature_band.tobytes()
            if signature_band_bytes not in hash_buckets_list[i]:
                hash_buckets_list[i][signature_band_bytes] = list()
            hash_buckets_list[i][signature_band_bytes].append(fh)
    
    # Check candidate pairs for similarity
    candidate_pairs = set()
    found = set()
    for hash_buckets in hash_buckets_list:
        for hash_bucket in hash_buckets.values():
            if len(hash_bucket) > 1:
                hash_bucket = sorted(hash_bucket)
                for i in range(len(hash_bucket)):
                    for j in range(i+1, len(hash_bucket)):
                        cpa = hash_bucket[i]
                        cpb = hash_bucket[j]
                        tup = tuple([cpa,cpb])
                        if not tup in found:
                            if is_valid(cpa,cpb):
                                candidate_pairs.add(tup)
                            found.add(tup)

    candidate_pairs_xor = list()
    for cpa, cpb in candidate_pairs:
        xor = np.bitwise_xor(np.unpackbits(signatures[cpa]),
                                np.unpackbits(signatures[cpb]))
        candidate_pairs_xor.append((cpa,cpb,xor))

    with Pool() as pool:
        candidate_pairs_sum = pool.map(bin_sum, candidate_pairs_xor)
        pool.close()
        pool.join()

    #Keep pairs that score above threshold score
    near_duplicates = list()
    _hash_size = HASH_SIZE**2
    for cpa, cpb, hd in candidate_pairs_sum:
        similarity = (_hash_size - hd) / _hash_size
        if similarity >= THRESHOLD:
            near_duplicates.append((cpa, cpb, similarity))
            
    # Sort near-duplicates by descending similarity and return
    if EVAL == 'quality':
            near_duplicates.sort(key=lambda x: x[2], reverse=True)
    elif EVAL == 'length':
        def calc_length(cpa, cpb):
            start_obj = datetime.strptime(cpa, '%H-%M-%S.%f')
            end_obj   = datetime.strptime(cpb, '%H-%M-%S.%f')
            return end_obj - start_obj
        near_duplicates.sort(key=lambda x: calc_length(x[0], x[1]), reverse=True)
    return near_duplicates

def prune_candidates(sims):
    def sort_candidates(x):
        return x[2], datetime.strptime(x[1], '%H-%M-%S.%f') - datetime.strptime(x[0], '%H-%M-%S.%f')
    
    sims.sort(key=sort_candidates, reverse=True)
    segments = list()
    final_sims = list()
    for s in sims:
        seg_start = datetime.strptime(s[0], '%H-%M-%S.%f')
        seg_end = datetime.strptime(s[1], '%H-%M-%S.%f')
        valid = True
        for seg in segments:
            if (seg_start >= seg[0] and seg_start <= seg[1]) or \
            (seg_end >= seg[0] and seg_end <= seg[1]) or \
            (seg_start <= seg[0] and seg_end >= seg[1]):
                valid = False
                break
        if valid:
            segments.append((seg_start,seg_end))
            final_sims.append(s)
    
    return final_sims

def video_to_gifs(gif_folder, final_sims):
    for i,s in enumerate(final_sims):
        clip = full_video.subclip(s[0].replace('-',':'),s[1].replace('-',':'))
        clip.write_gif(os.path.join(gif_folder, s[0].replace('-',':')+'-'+s[1].replace('-',':')+".gif"), verbose=False, logger=None)

def process_clip(split, gif_folder):
    signatures, fd_to_idx, frame_list = calculate_signatures(split)
    sims = find_similarity(signatures, fd_to_idx, frame_list)
    final_sims = prune_candidates(sims)
    video_to_gifs(gif_folder, final_sims)
    return len(final_sims)

def process_video(args):
    global full_video
    global MIN_LENGTH
    global MAX_LENGTH
    global THRESHOLD
    global EVAL
    global HASH_FUNC
    global FPS
    global HASH_SIZE
    global BANDS

    video_file = args.filepath
    assert (os.path.exists(video_file)), 'Invalid video_path: video file not found'
    mime = magic.Magic(mime=True)
    filename = mime.from_file(video_file)
    if filename.find('video') == -1:
        assert False, 'Invalid video_path: file is not a video'
    result_folder = args.result_folder
    assert (os.path.exists(result_folder)), 'Invalid result_folder: given result_folder path does not exist'

    MIN_LENGTH = args.min_len
    MAX_LENGTH = args.max_len
    assert ((MIN_LENGTH <= MAX_LENGTH) and (MIN_LENGTH >= 0.1) and (MAX_LENGTH <= 30)), 'Invalid max and min length'

    THRESHOLD = args.threshold
    assert ((THRESHOLD >= 0) and (THRESHOLD <= 100)), 'Invalid threshold value.'
    THRESHOLD /= 100.0
    EVAL = args.eval.lower().strip()
    assert (EVAL in ['quality','length']), 'Invalid eval: must be either "quality" or "length".'
    hash_name = args.hash_func.lower().strip()
    assert (hash_name in ['ahash','phash','dhash','whash']), 'Invalid hash_func: must be "ahash", "phash", "dhash", or "whash".'
    if hash_name == 'ahash':
        HASH_FUNC = imagehash.ahash
    elif hash_name == 'phash':
        HASH_FUNC = imagehash.phash
    elif hash_name == 'dhash':
        HASH_FUNC = imagehash.dhash
    elif hash_name == 'whash':
        HASH_FUNC = imagehash.whash


    CLIP_LENGTH = args.clip
    assert (CLIP_LENGTH > (MAX_LENGTH + MIN_LENGTH)), 'Invalid clip: must be greater than (max + min).'
    FPS = args.fps
    HASH_SIZE = args.hash
    BANDS = args.bands

    min_td = timedelta(seconds=MIN_LENGTH)
    max_td = timedelta(seconds=MAX_LENGTH)
    
    #global full_video
    full_video = VideoFileClip(video_file)
    duration = full_video.duration
    start = args.start
    end = args.end
    assert (((end == -1) or (start < end)) and (start < duration)), 'Invalid video selection: selected start and end are invalid.'
    end = end if end != -1 and end <= duration else duration

    splits = list()
    i = start
    while i < end:
        if i + CLIP_LENGTH < end:
            splits.append((i,i + CLIP_LENGTH))
            i += CLIP_LENGTH - MAX_LENGTH
        else:
            splits.append((i,end))
            i = end
    
    total = 0
    for i in (pbar := tqdm(range(len(splits)), desc=f'{total} Loops Found')):
        pbar.set_description(f'{total} Loops Found', refresh=True)
        s = splits[i]
        total += process_clip(s, result_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finds fragments of a video that appear to repeat or "loop" and creates gifs of them.')
    
    # Video Path; only required argument
    parser.add_argument('filepath', type=str,
    help='Video file with path included.')

    # Folder where resulting gif should be saved
    parser.add_argument('-result', dest='result_folder', default='./gif_output' , required=False, type=str,
    help='Folder to store resulting gifs, if found, for this run. Deafult is "./gif_output".')

    # Gif length parameters
    parser.add_argument('-min', dest='min_len', default=0.5, required=False, type=float,
    help='Minimum length of resulting gif in seconds. Default is 0.5, minimum is 0.01.')
    parser.add_argument('-max', dest='max_len', default=10.0, required=False, type=float,
    help='Maximun length of resulting gif in seconds. Default is 10.')

    # Video section parameters
    parser.add_argument('-start', dest='start', default=0, required=False, type=float,
    help='When in the video to start looking for loops in seconds. Default is 0.')
    parser.add_argument('-end', dest='end', default=-1, required=False, type=float,
    help='When in the video to stop looking for gifs in seconds or -1 for end of video. Default is -1.')

    # Loop scoring parameters
    parser.add_argument('-threshold', dest='threshold', default=85, required=False, type=float,
    help='Minimum threshold score a gif must attain to qualitfy as a valid loop. Default is 85.')
    parser.add_argument('-eval', dest='eval', default='quality', required=False, type=str,
    help='Metric used to order gifs when generating results (results cannot overlap). Can be either "quality" or "length". Default is "qualify".')
    parser.add_argument('-hash_func', dest='hash_func', default='dhash', required=False, type=str,
    help='Hashing function to use when scoring potential results. Can be "ahash", "phash", "dhash", or "whash". Default is dhash. More info on these functions can be found here: https://github.com/JohannesBuchner/imagehash')
    
    # Advanced performance parameters; should probably only be changed if there are memory errors
    parser.add_argument('-clip', dest='clip', default=60, required=False, type=float,
    help='Length each section of the video will be broken up into when being processed. Should only be changed if there are memory issues. Default is 60.')
    parser.add_argument('-fps', dest='fps', default=100, required=False, type=int,
    help='Number of frames per seonds that will be reviewed for potential loops. Default in 100.')
    parser.add_argument('-hash', dest='hash', default=64, required=False, type=int,
    help='Size that we resize the image to when using our hash function. Default is 64.')
    parser.add_argument('-bands', dest='bands', default=32, required=False, type=int,
    help='Number of potential bands an image can fall into based on its hash value. Default it 32.')

    args = parser.parse_args()
    process_video(args)
