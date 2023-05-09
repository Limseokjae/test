from bs4 import BeautifulSoup
import re
import yaml
from urllib import parse
from crawler.naver.naver import get_tags, scrape_blog

def get_blog_content(posturl):
    blogid, postno = posturl.split("/")[-2:]

    response = scrape_blog(posturl)
    soup = BeautifulSoup(response.content, "html.parser")

    parent_div = soup.find('div', {'class': 'se-main-container'})
    children_div = parent_div.findChildren('div', {'class': re.compile("^se-section")})

    
    img_text_dic = {}
    cont_idx = 0
    for child_div in children_div:
        is_text = 'se-section-text' in child_div['class']
        is_image = 'se-section-image' in child_div['class']
        if is_text:
            texts = []
            for child_paragraph in child_div.find_all('p', {'class': re.compile("^se-text-paragraph")}):
                if child_paragraph.text != '\u200b':
                    texts.append(child_paragraph.text)
            img_text_dic[cont_idx] = {'type':'text', 'content':texts}
            cont_idx += 1

        if is_image:
            img_tags = child_div.find_all("img", {'src': True, 'data-width': True, 'data-height': True, 'alt': True})
            img_urls = []
            for img_tag in img_tags:
                img_url = img_tag['data-lazy-src']
                img_urls.append(img_url)

            img_text_dic[cont_idx] = {'type':'image', 'content':img_urls}
            cont_idx += 1
                

    blogid, postno = posturl.split("/")[-2:]
    tag_response = get_tags(blogid, postno)
    tags = parse.unquote(tag_response.json()['taglist'][0]['tagName']).split(",")

    data_dic = {
        'blog_url': posturl,
        'tags': tags,
        'post_contents': img_text_dic
    }

    return data_dic, postno

if __name__ == "__main__":

    posturl = "https://blog.naver.com/yee1036/223090678033"
    postdata, postno = get_blog_content(posturl)

    with open(f"./{postno}.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(postdata, f, allow_unicode=True)