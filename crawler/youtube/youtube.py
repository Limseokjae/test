import os
import argparse
import pandas as pd
from tqdm import tqdm
from pytube import YouTube


def main():
    parser = argparse.ArgumentParser(description='Youtube')
    parser.add_argument('-video', '--save_video', action='store_true')
    parser.add_argument('-meta', '--save_meta', action='store_true')
    args = parser.parse_args()
    data = pd.read_excel('/home/work/som2/R_test/data_urls_1000.xlsx')
    done = set()
    error = set()
    pbar = tqdm(range(len(data)))

    save_video_dir = 'butter_video'  # 영상 download할 디렉토리
    save_meta_dir = 'butter_meta'  # 영상 download할 디렉토리
    os.makedirs(save_video_dir, exist_ok=True)
    os.makedirs(save_meta_dir, exist_ok=True)
    print('=======')
    for i in pbar:
        url = data.iloc[i].url
        if url not in done:  # 진행하지 않은 url만 download 시도
            try:
                yt = YouTube(url)
                if args.save_video:
                    yt.streams.filter(progressive=False).order_by('resolution').desc().first().download(output_path=save_video_dir)
                if args.save_meta:
                    with open(os.path.join(save_meta_dir, f'{yt.title}.txt'), 'w') as f:
                        f.write(yt.description)
            except:
                error.add(url)
                print(url)

            done.add(url)  # 진행한 url 체크 
        pbar.set_description(f'{len(done)}')
        if i:
            break
    print(f'Done!!! -> {len(done)}')
    print(f'Error!!! -> {len(error)}')

    # Download 실패한 url들 text 파일로 저장
    with open('error.txt', 'w') as f:
        for e in error:
            f.write(f"{e}\n")
    

if __name__ == '__main__':
    main()
