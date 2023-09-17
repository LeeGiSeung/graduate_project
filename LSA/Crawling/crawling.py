# crawler_naver blog

### Step 0. 준비
import sys    # 시스템
import os     # 시스템
import re

import pandas as pd    # 판다스 : 데이터분석 라이브러리
import numpy as np     # 넘파이 : 숫자, 행렬 데이터 라이브러리

from bs4 import BeautifulSoup     # html 데이터 전처리
from selenium import webdriver    # 웹 브라우저 자동화
import time                       # 시간 지연
from tqdm import tqdm_notebook    # 진행상황 표시
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless")  # Headless 모드 활성화
chrome_options.add_argument("--disable-gpu")  # GPU 가속 비활성화 (필요한 경우)

# python 버전 확인
content_list = []
# 판다스 버전 확인
pd.__version__

### Step 1. 크롤링할 블로그 url 수집
# 검색어
keyword1 = input("1.크롤링할 키워드를 입력하세요: ")

# 크롬 웹브라우저 실행
ch_driver = r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver"

driver = webdriver.Chrome(ch_driver, options=chrome_options)

# 사이트 주소
driver.get("http://www.naver.com")
time.sleep(2)

# 검색창에 '검색어' 검색
element = driver.find_element(By.ID, "query")
element.send_keys(keyword1)
element.submit()
time.sleep(1)

# 'VIEW' 클릭

driver.find_element(By.LINK_TEXT, "VIEW").click()

# '블로그' 클릭
driver.find_element(By.LINK_TEXT, "블로그").click()
time.sleep(1)

# '옵션' 클릭
driver.find_element(By.LINK_TEXT, "옵션").click()
time.sleep(0.5)

driver.find_element(By.LINK_TEXT, "1년").click()



# 스크롤 다운
def scroll_down(driver):
    driver.execute_script("window.scrollTo(0, 99999999)")
    time.sleep(1)

# n: 스크롤할 횟수 설정
n = 500
i = 0
while i < n:
    scroll_down(driver)
    i = i+1

# 블로그 글 url들 수집
url_list = []
title_list = []
article_raw = driver.find_elements(By.CLASS_NAME, "api_txt_lines.total_tit")

# 크롤링한 url 정제 시작
for article in article_raw:
    url = article.get_attribute('href')   
    url_list.append(url)

time.sleep(1)
    
# 제목 크롤링 시작    
for article in article_raw:
    title = article.text
    title_list.append(title)

print("url개수: ", len(url_list))
print("title개수: ", len(title_list))

df = pd.DataFrame({'url':url_list, 'title':title_list})
df

# url_list 저장
df.to_excel("저장된1 url.xlsx")

### Step 2. 블로그 내용 크롤링
import sys
import os

import pandas as pd
import numpy as np

# "url_list.csv" 불러오기
# 수정해야할거
url_load = pd.read_excel("저장된1 url.xlsx")

num_list = len(url_load)

print("url리스트 갯수 : "+str(num_list))
url_load

data_dict = {}    # 전체 크롤링 데이터를 담을 그릇

number = num_list    # 수집할 글 갯수

# 수집한 url 돌면서 데이터 수집
for i in tqdm_notebook(range(0, number)):
    # 글 띄우기
    print(i)
    url = url_load["url"][i]
    driver = webdriver.Chrome(r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver", options=chrome_options)
    driver.get(url)   # 글 띄우기
    
    # 크롤링
    
# 크롤링

    try : 
        # 글의 iframe 접근
        driver.switch_to.frame("mainFrame")

        target_info = {}  # 개별 블로그 내용을 담을 딕셔너리 생성

        # 제목 크롤링
        overlays = ".se-module.se-module-text.se-title-text"                        
        tit = driver.find_element(By.CSS_SELECTOR, overlays)
        
        title = tit.text

        # 글쓴이 크롤링
        overlays = ".nick"                                 
        nick = driver.find_element(By.CSS_SELECTOR, overlays)
        nickname = nick.text

        # 날짜 크롤링
        overlays = ".se_publishDate.pcol2"    
        date = driver.find_element(By.CSS_SELECTOR, overlays)

        datetime = date.text

        # 내용 크롤링
        content_list = []  # content_list 초기화

        overlays = ".se-component.se-text.se-l-default"                                 
        contents = driver.find_elements(By.CSS_SELECTOR, overlays)  # 요소들을 가져옴

        # print("contents 실행")
        # print(contents)
        for content in contents:
            content_text = content.text  # 각 요소의 텍스트를 가져옴
            content_list.append(content_text)
            
        content_str = ' '.join(content_list)


        # 크롤링한 글은 target_info라는 딕셔너리에 담음
        target_info['title'] = title
        target_info['nickname'] = nickname
        target_info['datetime'] = datetime
        target_info['content'] = content_str
        target_info['label'] = 1

        # 각각의 글은 data_dict라는 딕셔너리에 담음
        data_dict[i] = target_info
        time.sleep(1)
        
        # 크롤링 성공하면 글 제목을 출력
        print("크롤링 성공", i, title)

        # 글 하나 크롤링 후 크롬 창 닫기
        driver.close()       
    
    # 에러나면 현재 크롬창을 닫고 다음 글(i+1)로 이동
    except:
        driver.close()
        time.sleep(1)
        continue

    
    # # 중간 저장
    # if i in [100, 500, 1000]:
    #     # 판다스로 만들기
    #     import pandas as pd
    #     result_df = pd.DataFrame.from_dict(data_dict, 'index')

    #     # 저장하기
    #     result_df.to_excel(f"result_{i}.xlsx", encoding='utf-8-sig')   # 각각의 반복마다 다른 파일명으로 저장
    #     time.sleep(3)


# print('수집한 글 갯수: ', len(data_dict))
# # print(data_dict)

# # 판다스화
result_df = pd.DataFrame.from_dict(data_dict, 'index')
# # 엑셀 저장
result_df['content'] = result_df['content'].astype(str)

# # 수정해야할거
# result_df.to_excel("활동적 특징.xlsx", encoding='utf-8')
result_df.to_csv("MBTI I 성향 특징 살펴보기2.csv", encoding='utf-8-sig', index=False)

