
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import quote


def delete_iframe(url):
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"}
    res = requests.get(url, headers=headers)
    response_2 = requests.get(url)
    soup = BeautifulSoup(response_2.content, "html.parser")
    url2 = soup.find_all("iframe")[0].get("src")
    src_url = "https://blog.naver.com/" + url2
    return src_url

def text_scraping(url):
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"}
    res = requests.get(url, headers=headers)
    res.raise_for_status() # 문제시 프로그램 종료
    soup = BeautifulSoup(res.text, "lxml") 
    if soup.find("div", attrs={"class":"se-main-container"}):
        text = soup.find("div", attrs={"class":"se-main-container"}).get_text()
        text = text.replace("\n","") #공백 제거
        return text

    elif soup.find("div", attrs={"id":"postViewArea"}):
        text = soup.find("div", attrs={"id":"postViewArea"}).get_text()
        text = text.replace("\n","") 
        return text
    else:
        return False
    
def scrape_blog(url):
    src_url = delete_iframe(url)
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"}
    res = requests.get(src_url, headers=headers)
    res.raise_for_status() # 문제시 프로그램 종료
    return res

def get_blog_list(keyword, countperpage, currentpage):
    url = "https://section.blog.naver.com/ajax/SearchList.naver"

    querystring = {"countPerPage":str(countperpage),"currentPage":str(currentpage),"endDate":"","keyword":keyword,"orderBy":"sim","startDate":"","type":"post"}

    payload = ""
    headers = {
        "authority": "section.blog.naver.com",
        "accept": "application/json, text/plain, */*",
        "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,tr;q=0.6,it;q=0.5",
        "referer": "https://section.blog.naver.com/Search/Post.naver?pageNo=1&rangeType=ALL&orderBy=sim&keyword=^%^EC^%^9A^%^94^%^EB^%^A6^%^AC",
        "sec-ch-ua": "^\^Chromium^^;v=^\^112^^, ^\^Google",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "^\^Windows^^",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    }

    response = requests.request("GET", url, data=payload, headers=headers, params=querystring)

    crawl_result = json.loads(",".join(response.text.split(",")[1:]))
    return crawl_result

def get_tags(blogid, postno):
    url = "https://blog.naver.com/BlogTagListInfo.naver"
    querystring = {"blogId":blogid,"logNoList":str(postno),"logType":"mylog"}
    payload = ""
    headers = {
        "authority": "blog.naver.com",
        "accept": "*/*",
        "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "charset": "utf-8",
        "content-type": "application/x-www-form-urlencoded; charset=utf-8",
        "sec-ch-ua": "^\^Chromium^^;v=^\^112^^, ^\^Google",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "^\^Windows^^",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    }

    response = requests.get(url, data=payload, headers=headers, params=querystring)
    return response