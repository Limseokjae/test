import os
import re
import sys
import cv2
import json
import argparse
import requests
import itertools

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

from pytube import YouTube
from pytube import extract

import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


SAVE_VIDEO_DIR = 'butter_video'  # 영상 download할 디렉토리
SAVE_META_DIR = 'butter_meta'
SAVE_CAPTION_DIR = 'butter_caption'
SAVE_FRAME_DIR = 'butter_frame'


def crawl_meta(video_id):

    soup = BeautifulSoup(requests.get(f'https://www.youtube.com/watch?v={video_id}').content)
    pattern = re.compile('(?<=shortDescription":").*(?=","isCrawlable)')
    description = pattern.findall(str(soup))[0].replace('\\n','\n')
    return description


def download(url, save_video, save_meta):
    try:
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)

        if save_video:
            video = yt.streams.filter(progressive=False).order_by('resolution').desc().first()
            video.download(output_path=SAVE_VIDEO_DIR, filename=f'{yt.video_id}.{video.subtype}')

            os.makedirs(os.path.join(SAVE_FRAME_DIR, yt.video_id), exist_ok=True)
            vidcap = cv2.VideoCapture(os.path.join(SAVE_VIDEO_DIR, f'{yt.video_id}.{video.subtype}'))
            fps = int(vidcap.get(cv2.CAP_PROP_FPS))
            success, image = vidcap.read()
            count = 0
            while success:
                if vidcap.get(cv2.CAP_PROP_POS_FRAMES) % fps == 0:
                    # save frame as PNG file
                    count += 1
                    cv2.imwrite(os.path.join(SAVE_FRAME_DIR, yt.video_id, f"frame_{count:05}.png"), image)
                success, image = vidcap.read()

        if save_meta:
            yt.metadata  # Meta 정보 업데이트

            # content = yt.initial_data['contents']['twoColumnWatchNextResults']['results']['results']['contents'][1]['videoSecondaryInfoRenderer']['attributedDescription']['content']
            content = crawl_meta(yt.video_id)
            with open(os.path.join(SAVE_META_DIR, f'{yt.video_id}.txt'), 'w') as f:
                f.write(content)

            try: 
                transcript = YouTubeTranscriptApi.get_transcript(yt.video_id, languages=['ko'])
                with open(os.path.join(SAVE_CAPTION_DIR, f'{yt.video_id}.json'), 'w') as f:
                    json.dump(transcript, f, indent=2, ensure_ascii=False)
            except youtube_transcript_api.TranscriptsDisabled as e:  # case: 자막 사용 불가
                return (url, 'caption')
            except youtube_transcript_api.NoTranscriptFound as e:  # case: 자동 번역
                transcript_list = YouTubeTranscriptApi.list_transcripts(yt.video_id)
                transcript = next(iter(transcript_list)).translate('ko').fetch()
                with open(os.path.join(SAVE_CAPTION_DIR, f'{yt.video_id}.json'), 'w') as f:
                    json.dump(transcript, f, indent=2, ensure_ascii=False)

    except Exception as e:
        # error.add((url, 'error'))
        return (url, 'error')

    return (url, 'done')


def main():
    parser = argparse.ArgumentParser(description='Youtube')
    parser.add_argument('-v', '--video', action='store_true')
    parser.add_argument('-m', '--meta', action='store_true')
    parser.add_argument('-a', '--all', action='store_true')
    args = parser.parse_args()

    data = pd.read_csv('urls.csv')
    done = set()
    error = set()

    pbar = tqdm(range(len(data)))

    if args.all:
        args.video = True
        args.meta = True
    
    if args.video:
        os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)
        os.makedirs(SAVE_FRAME_DIR, exist_ok=True)

    if args.meta:
        os.makedirs(SAVE_META_DIR, exist_ok=True)
        os.makedirs(SAVE_CAPTION_DIR, exist_ok=True)

    with ProcessPoolExecutor(4) as executor:
        for url, code in executor.map(download, 
                                      data.url.tolist(), 
                                      itertools.repeat(args.video, len(data)),
                                      itertools.repeat(args.meta, len(data))):
            if code != 'done':
                error.add((url, code))
            else:
                done.add((url, code))
            pbar.update(1)

    print(f'Done!!! -> {len(done)}')
    print(f'Error!!! -> {len(error)}')

    # Download 실패한 url들 text 파일로 저장
    with open('error.txt', 'w') as f:
        for e in error:
            f.write(f"{e}\n")
    

if __name__ == '__main__':
    main()
