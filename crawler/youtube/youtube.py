import os
import re
import sys
import json
import argparse
import requests

import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

from pytube import YouTube
from pytube import extract

import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


def crawl_meta(video_id):

    soup = BeautifulSoup(requests.get(f'https://www.youtube.com/watch?v={video_id}').content)
    pattern = re.compile('(?<=shortDescription":").*(?=","isCrawlable)')
    description = pattern.findall(str(soup))[0].replace('\\n','\n')
    return description


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
        save_video_dir = 'butter_video'  # 영상 download할 디렉토리
        os.makedirs(save_video_dir, exist_ok=True)

    if args.meta:
        save_meta_dir = 'butter_meta'
        save_caption_dir = 'butter_caption'
        os.makedirs(save_meta_dir, exist_ok=True)
        os.makedirs(save_caption_dir, exist_ok=True)

        formatter = TextFormatter()

    for i in pbar:
        url = data.iloc[i].url
        try:
            yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)

            if args.video:
                video = yt.streams.filter(progressive=False).order_by('resolution').desc().first()
                video.download(output_path=save_video_dir, filename=f'{yt.video_id}.{video.subtype}')

            if args.meta:
                yt.metadata  # Meta 정보 업데이트

                # content = yt.initial_data['contents']['twoColumnWatchNextResults']['results']['results']['contents'][1]['videoSecondaryInfoRenderer']['attributedDescription']['content']
                content = crawl_meta(yt.video_id)
                with open(os.path.join(save_meta_dir, f'{yt.video_id}.txt'), 'w') as f:
                    f.write(content)

                try: 
                    transcript = YouTubeTranscriptApi.get_transcript(yt.video_id, languages=['ko'])
                    with open(os.path.join(save_caption_dir, f'{yt.video_id}.json'), 'w') as f:
                        json.dump(transcript, f, indent=2, ensure_ascii=False)
                except youtube_transcript_api.TranscriptsDisabled as e:  # case: 자막 사용 불가
                    error.add((url, 'caption'))
                except youtube_transcript_api.NoTranscriptFound as e:  # case: 자동 번역
                    transcript_list = YouTubeTranscriptApi.list_transcripts(yt.video_id)
                    transcript = next(iter(transcript_list)).translate('ko').fetch()
                    with open(os.path.join(save_caption_dir, f'{yt.video_id}.json'), 'w') as f:
                        json.dump(transcript, f, indent=2, ensure_ascii=False)

                # caption = formatter.format_transcript(transcript)
                # with open(os.path.join(save_caption_dir, f'{yt.title}.txt'), 'w') as f:
                #     f.write(caption)

            done.add(url)  # 진행한 url 체크 
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            error.add((url, 'error'))

        pbar.set_description(f'Done: {len(done)} | Error: {len(error)}')

    print(f'Done!!! -> {len(done)}')
    print(f'Error!!! -> {len(error)}')

    # Download 실패한 url들 text 파일로 저장
    with open('error.txt', 'w') as f:
        for e in error:
            f.write(f"{e}\n")
    

if __name__ == '__main__':
    main()
