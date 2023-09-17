# crawler_naver blog

### Step 0. 준비
import sys    # 시스템
import os     # 시스템
import re
import http.server
import socketserver
import json
from pymongo import MongoClient
import requests
import threading
import gensim
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from konlpy.tag import Komoran
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from pymongo.cursor import CursorType

import pandas as pd    # 판다스 : 데이터분석 라이브러리
import numpy as np     # 넘파이 : 숫자, 행렬 데이터 라이브러리

from bs4 import BeautifulSoup     # html 데이터 전처리
from selenium import webdriver    # 웹 브라우저 자동화
import time                       # 시간 지연
from tqdm import tqdm_notebook    # 진행상황 표시
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
komoran = Komoran()
chrome_options = Options()
chrome_options.add_argument("--headless")  # Headless 모드 활성화
chrome_options.add_argument("--disable-gpu")  # GPU 가속 비활성화 (필요한 경우)

def process_and_store_data(setntence_input):

    # 어절로 나누기
    words = setntence_input.split()

    # 명사와 동사 추출
    selected_words = []
    for word in words:
        pos_tags = komoran.pos(word)
        for token, pos in pos_tags:
            if pos in ['NNG', 'NNP', 'VV', 'VA']:  # 명사와 동사의 품사 태그
                selected_words.append(token)

    dictionary = corpora.Dictionary([selected_words])

    tfidf_vectorizer = TfidfVectorizer(vocabulary=dictionary.token2id)
    tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(selected_words)])

    # CSR이란? 행렬이 커지면 커질수록 중복이 많아지는데, 이 중복을 과감히 제거하고, 행이 시작하는 지점을 표시해주는 방식이다
    # TF-IDF 행렬을 CSR 포맷으로 변환
    tfidf_matrix_csr = csr_matrix(tfidf_matrix.transpose())

    # CSR 포맷의 행렬을 Gensim Corpus 형식으로 변환하여 MmCorpus 생성
    tfidf_corpus = gensim.matutils.Sparse2Corpus(tfidf_matrix_csr)

    # 토픽 모델링실행
    num_topics = 1
    # passes는 반복수 높을 수록 좋은 결과를 얻지만 시간이 오래 걸릴 수 있고 오버피팅 될 수 있음
    lda_model = LdaModel(corpus=tfidf_corpus, num_topics=num_topics, id2word=dictionary, passes=1)

    # 토픽과 각 토픽의 단어 및 확률 출력한다
    # nun_words 한 토픽당 나올 단어
    topics = lda_model.print_topics(num_topics=num_topics, num_words=3)
    
    tokenized_documents = [doc.split() for doc in selected_words]

    # 생성한 LDA 모델을 사용하여 CoherenceModel 객체 생성
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v', processes=1)

    print("답변 내용 : ",setntence_input)
    print("토픽 모델링 결과 : ",topics)

    # 일관성 점수 계산
    coherence_score = coherence_model_lda.get_coherence()
    print("Coherence Score:", coherence_score)

# python 버전 확인
content_list = []
# 판다스 버전 확인
pd.__version__

### Step 1. 크롤링할 블로그 url 수집
# 검색어
keyword1 = input("1.크롤링할 키워드를 입력하세요: ")
# keyword2 = input("2.제외할 첫번째 키워드를 입력하세요: ")
# keyword3 = input("3.제외할 두번째 키워드를 입력하세요: ")

# 크롬 웹브라우저 실행
ch_driver = r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver"

driver = webdriver.Chrome(ch_driver, options=chrome_options)

# 사이트 주소
driver.get("http://www.naver.com")
time.sleep(2)

# 검색창에 '검색어' 검색
# element = driver.find_element(By.ID, "query")
element = driver.find_element(By.ID, "query")
element.send_keys(keyword1)
element.submit()
time.sleep(1)

# 'VIEW' 클릭

driver.find_element(By.LINK_TEXT, "지식iN").click()

# '옵션' 클릭
driver.find_element(By.LINK_TEXT, "옵션").click()
time.sleep(1)

# '1년' 클릭
driver.find_element(By.LINK_TEXT, "1년").click()
time.sleep(0.5)

# 스크롤 다운
def scroll_down(driver):
    driver.execute_script("window.scrollTo(0, 99999999)")
    time.sleep(1)

# n: 스크롤할 횟수 설정
n = 0
i = 0
while i < n:
    scroll_down(driver)
    i = i+1

# 블로그 글 url들 수집
url_list = []
title_list = []
article_raw = driver.find_elements(By.CLASS_NAME, "question_text")

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

print("title리스트", title_list)

# url_list 저장 
df.to_excel("네이버지식인 url.xlsx")
# 여기까진 문제없음
### Step 2. 블로그 내용 크롤링
import sys
import os

import pandas as pd
import numpy as np

# 수정해야할거
url_load = pd.read_excel("네이버지식인 url.xlsx")

num_list = len(url_load)

print("url리스트 갯수 : "+str(num_list))
url_load

data_dict = {}    # 전체 크롤링 데이터를 담을 그릇

number = num_list    # 수집할 글 갯수

# 수집한 url 돌면서 데이터 수집
for i in tqdm_notebook(range(0, number)):
    # 글 띄우기
    print("순서",i)
    url = url_load["url"][i]
    driver = webdriver.Chrome(r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver", options=chrome_options)
    driver.get(url)   # 글 띄우기
    
# 크롤링

    target_info = {}  # 개별 블로그 내용을 담을 딕셔너리 생성

        # 질문 제목 크롤링
    overlays = "div.c-heading__title .title"                        
    tit = driver.find_element(By.CSS_SELECTOR, overlays)
    title = tit.text
    print("질문 제목 크롤링",title)
    print()

        # 질문 내용 크롤링
    try:
        overlays = ".c-heading__content"                                 
        cont = driver.find_element(By.CSS_SELECTOR, overlays)
        content = cont.text
        print("질문 내용 : ",content)
        print()
    except:
        print("질문 내용은 없습니다.")

        # 답변 크롤링

    try:    
        overlays = ".c-heading-answer__content"    
        ans = driver.find_element(By.CSS_SELECTOR, overlays)
        answer = ans.text
        print("질문 답변 : ",answer)
        print()
        process_and_store_data(answer)
        print()
    except:
        print("답변 내용은 없습니다.")

        # 크롤링한 글은 target_info라는 딕셔너리에 담음
    target_info['title'] = title
    target_info['content'] = content
    target_info['answer'] = answer

        # 각각의 글은 data_dict라는 딕셔너리에 담음
    data_dict[i] = target_info
    time.sleep(1)
        
        # 크롤링 성공하면 글 제목을 출력
    # print("크롤링 성공", i, title)

        # 글 하나 크롤링 후 크롬 창 닫기
    driver.close()    
    


