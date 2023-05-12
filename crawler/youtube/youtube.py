import os
import argparse
import pandas as pd
from tqdm import tqdm

from pytube import YouTube
from pytube import extract

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


def main():
    parser = argparse.ArgumentParser(description='Youtube')
    parser.add_argument('-video', '--save_video', action='store_true')
    parser.add_argument('-meta', '--save_meta', action='store_true')
    args = parser.parse_args()

    data = pd.read_excel('data_urls_1000.xlsx')
    done = set()
    error = set()

    pbar = tqdm(range(len(data)))
    
    if args.save_video:
        save_video_dir = 'butter_video'  # 영상 download할 디렉토리
        os.makedirs(save_video_dir, exist_ok=True)

    if args.save_meta:
        save_meta_dir = 'butter_meta'
        save_caption_dir = 'butter_caption'
        os.makedirs(save_meta_dir, exist_ok=True)
        os.makedirs(save_caption_dir, exist_ok=True)

        formatter = TextFormatter()

    for i in pbar:
        url = data.iloc[i].url
        if url not in done:  # 진행하지 않은 url만 download 시도
            try:
                yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)

                if args.save_video:
                    yt.streams.filter(progressive=False).order_by('resolution').desc().first().download(output_path=save_video_dir)
                    done.add(url)  # 진행한 url 체크 

                if args.save_meta:
                    yt.metadata

                    content = yt.initial_data['contents']['twoColumnWatchNextResults']['results']['results']['contents'][1]['videoSecondaryInfoRenderer']['attributedDescription']['content']
                    with open(os.path.join(save_meta_dir, f'{yt.title}.txt'), 'w') as f:
                        f.write(content)

                    transcript = YouTubeTranscriptApi.get_transcript(yt.video_id, languages=['ko'])
                    caption = formatter.format_transcript(transcript)
                    with open(os.path.join(save_caption_dir, f'{yt.title}.txt'), 'w') as f:
                        f.write(caption)
                    done.add(url)  # 진행한 url 체크 
            except:
                error.add(url)
                pbar.set_description(f'')
                # print(url)

        pbar.set_description(f'Done: {len(done)} | Error: {len(error)}')

    print(f'Done!!! -> {len(done)}')
    print(f'Error!!! -> {len(error)}')

    # Download 실패한 url들 text 파일로 저장
    with open('error.txt', 'w') as f:
        for e in error:
            f.write(f"{e}\n")
    

if __name__ == '__main__':
    main()
