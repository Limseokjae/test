import os
import urllib
import tempfile
from crawler.naver.main import get_blog_content
from model.sam_vit import SAMVIT
from text.text_processor import get_info
import cmd_args

def main(tmp_dir):
    posturl = "https://blog.naver.com/peace8012/223098210044"
    postdata, postno = get_blog_content(posturl)
    idx = 0
    SamVitModel = SAMVIT()
    SamVitModel.load_model()
    txt = []
    print("blog_url", postdata['blog_url'])
    print("post_contents")
    for idx in postdata['post_contents'].keys():
        content_type = postdata["post_contents"][idx]['type']
        contents = postdata["post_contents"][idx]['content']
        if len(contents) < 1:
            continue
        if content_type == 'text':
            texts = postdata["post_contents"][idx]['content']
            txt.append(texts)
            # print(content_type, texts)
        elif content_type == 'image':
            img_url = postdata["post_contents"][idx]['content'][0]
            ext = img_url.split(".")[-1].split("?")[0]
            img_name = f"{idx:05d}.{ext}"
            save_path = os.path.join(tmp_dir, img_name)
            urllib.request.urlretrieve(img_url, save_path)
            # print(content_type, save_path)

            # 이미지 읽어서 모델돌리고 등등 처리
            SamVitModel.run_model(save_path)
            # image = cv2.imread(save_path)x1
        else:
            pass

    ## text data
     
    openai_keys=open('./api.txt','r').read().splitlines()[0]
    response = get_info(openai_keys,txt)

    ## post processing

    result_df = {"url": [],"title":[],"frame_number":[],"xywh":[],"top1":[],"top1_prob":[],
                "top2":[],"top2_prob":[],"top3":[],"top3_prob":[],
                "top4":[],"top4_prob":[],"top5":[],"top5_prob":[]
                }

if __name__ == "__main__":

    args, _ = cmd_args.parser.parse_known_args()
    print(args)     
    
    if args.keep_files:
        tmp_dir = tempfile.mkdtemp(dir=os.getcwd())
        main(tmp_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            main(tmp_dir)

