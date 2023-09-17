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
komoran = Komoran()

# MongoDB 연결
client = MongoClient('localhost', 27017)
db = client['VScodeDB']
collection = db['DemoDatabase']

class MyRequestHandler(http.server.BaseHTTPRequestHandler):
    def _set_response(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_response()
        name = self.path.split('/')[-1]
        result = collection.find_one({'name': name})
        if result:
            self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            self.wfile.write(json.dumps({'message': 'Data not found'}).encode('utf-8'))

    def do_POST(self):
        self._set_response()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        collection.insert_one(data)
        self.wfile.write(json.dumps({'message': 'Data inserted successfully'}).encode('utf-8'))

def run(server_class=socketserver.ThreadingTCPServer, handler_class=MyRequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}\n")

    # 서버를 별도의 스레드에서 실행
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.start()

# 입력 받은 문장 LDA 구하기
def process_and_store_data(setntence_input, user_input):

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

    # TF-IDF를 사용한 전처리
    # tfidf_vectorizer = TfidfVectorizer(vocabulary=dictionary.token2id)
    # tfidf_matrix = tfidf_vectorizer.fit_transform(selected_words)

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

    topic_dic = {
        'id' : user_input,
        'original_data' : setntence_input,
        'LAD_data' : topics
    }
    
    tokenized_documents = [doc.split() for doc in selected_words]

    # 생성한 LDA 모델을 사용하여 CoherenceModel 객체 생성
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v', processes=1)

    # 일관성 점수 계산
    coherence_score = coherence_model_lda.get_coherence()
    print("Coherence Score:", coherence_score)

    return(topic_dic)

# 지식인 답변 검색 토픽 5개 구하기
def process_intellectual(setntence_input):

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

    # TF-IDF를 사용한 전처리
    # tfidf_vectorizer = TfidfVectorizer(vocabulary=dictionary.token2id)
    # tfidf_matrix = tfidf_vectorizer.fit_transform(selected_words)

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
    topics = lda_model.print_topics(num_topics=num_topics, num_words=5)
    
    tokenized_documents = [doc.split() for doc in selected_words]

    # 생성한 LDA 모델을 사용하여 CoherenceModel 객체 생성
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v', processes=1)

    # 일관성 점수 계산
    coherence_score = coherence_model_lda.get_coherence()
    print("Coherence Score:", coherence_score)

    return(topics)

# 지식인 답변 검색(5개)
def Find_Anser(insert_keyword):
    import re

    ### Step 1. 크롤링할 블로그 url 수집
    # 검색어
    # 만약 검색했는데 검색된 글이 아무것도 없다면 오류가 나옴
    print("########################")
    print("지식인 검색 : ",insert_keyword)
    print("########################")
    keyword1 = process_intellectual_result(insert_keyword)
    # 크롬 웹브라우저 실행
    ch_driver = r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver"

    driver = webdriver.Chrome(ch_driver, options=chrome_options)

    # 사이트 주소
    driver.get("http://www.naver.com")
    time.sleep(2)

    # 검색창에 '검색어' 검색
    # element = driver.find_element(By.ID, "query")
    element = driver.find_element(By.ID, "query")
    # element.send_keys(str(re.sub(r'[^a-zA-Z가-힣\s]', '', keyword1)))

    filtered_keyword = re.sub(r'[^a-zA-Z가-힣\s]', '', str(keyword1))
    print("필터링 지식인 검색 키워드",filtered_keyword)
    if isinstance(filtered_keyword, str):
        element.send_keys(filtered_keyword)

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

    ### Step 2. 블로그 내용 크롤링
    import sys
    import os
    import pandas as pd
    import numpy as np

    num_list = len(url_list)

    print("url리스트 갯수 : ",num_list)

    data_dict = {}    # 전체 크롤링 데이터를 담을 그릇

    number = num_list    # 수집할 글 갯수
    answer_result = ""
    # 수집한 url 돌면서 데이터 수집
    for i in range(len(url_list)):
        # 글 띄우기
        print("순서",i)
        url = url_list[i]
        driver = webdriver.Chrome(r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver", options=chrome_options)
        driver.get(url)   # 글 띄우기
        
    # 크롤링

            # 질문 제목 크롤링
        try:
            overlays = "div.c-heading__title .title"                        
            tit = driver.find_element(By.CSS_SELECTOR, overlays)
            title = tit.text
            print("질문 제목 크롤링",title)
            print()
        except:
            print("양식에 맞지않아 통과합니다.")

            # 질문 내용 크롤링
        try:
            overlays = ".c-heading__content"                                 
            cont = driver.find_element(By.CSS_SELECTOR, overlays)
            content = cont.text
            # print("질문 내용 : ",content)
            print()
        except:
            print("질문 내용은 없습니다.")

            # 답변 크롤링

        try:    
            overlays = ".c-heading-answer__content"    
            ans = driver.find_element(By.CSS_SELECTOR, overlays)
            answer = ans.text
            # print("질문 답변 : ",answer)
            print()
            answer_result = answer_result + answer
            print()
        except:
            print("답변 내용은 없습니다.")
    topic = process_intellectual(answer_result)
    print("지식인 답변 토픽 : ",topic)
    Find_Final_Anser(topic)
    driver.close() 

# 지식인 최종 답변 토픽 구하기
def process_intellectual_result(keyword):

    # 어절로 나누기
    words = keyword.split()

    # 명사와 동사 추출
    selected_words = []
    for word in words:
        pos_tags = komoran.pos(word)
        for token, pos in pos_tags:
            if pos in ['NNG', 'NNP', 'VV', 'VA']:  # 명사와 동사의 품사 태그
                selected_words.append(token)

    dictionary = corpora.Dictionary([selected_words])

    # TF-IDF를 사용한 전처리
    # tfidf_vectorizer = TfidfVectorizer(vocabulary=dictionary.token2id)
    # tfidf_matrix = tfidf_vectorizer.fit_transform(selected_words)

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
    topics = lda_model.print_topics(num_topics=num_topics, num_words=5)
    
    tokenized_documents = [doc.split() for doc in selected_words]

    # 생성한 LDA 모델을 사용하여 CoherenceModel 객체 생성
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v', processes=1)

    # 일관성 점수 계산
    coherence_score = coherence_model_lda.get_coherence()
    print("Coherence Score:", coherence_score)

    return(topics)

# 지식인 최종 답변 도출
def Find_Final_Anser(topics):
    import re

    ### Step 1. 크롤링할 블로그 url 수집
    # 검색어
    # 만약 검색했는데 검색된 글이 아무것도 없다면 오류가 나옴
    print("########################")
    print("최종 추천(네이버 지식in)")
    # 크롬 웹브라우저 실행
    ch_driver = r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver"

    driver = webdriver.Chrome(ch_driver, options=chrome_options)

    # 사이트 주소
    driver.get("http://www.naver.com")
    time.sleep(2)

    # 검색창에 '검색어' 검색
    # element = driver.find_element(By.ID, "query")
    element = driver.find_element(By.ID, "query")
    # element.send_keys(str(re.sub(r'[^a-zA-Z가-힣\s]', '', keyword1)))

    filtered_keyword = re.sub(r'[^a-zA-Z가-힣\s]', '', str(topics))
    # print("필터링 지식인 검색 키워드",filtered_keyword)
    if isinstance(filtered_keyword, str):
        element.send_keys(filtered_keyword)

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

    ### Step 2. 블로그 내용 크롤링
    import sys
    import os
    import pandas as pd
    import numpy as np

    num_list = len(url_list)

    print("url리스트 갯수 : ",num_list)

    data_dict = {}    # 전체 크롤링 데이터를 담을 그릇

    number = num_list    # 수집할 글 갯수
    answer_result = ""
    # 수집한 url 돌면서 데이터 수집
    for i in range(1):
        # 글 띄우기
        url = url_list[i]
        driver = webdriver.Chrome(r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver", options=chrome_options)
        driver.get(url)   # 글 띄우기
        
    # 크롤링

            # 질문 제목 크롤링
        try:
            overlays = "div.c-heading__title .title"                        
            tit = driver.find_element(By.CSS_SELECTOR, overlays)
            title = tit.text
            # print("질문 제목 크롤링",title)
            print()
        except:
            print("양식에 맞지않아 통과합니다.")

            # 질문 내용 크롤링
        try:
            overlays = ".c-heading__content"                                 
            cont = driver.find_element(By.CSS_SELECTOR, overlays)
            content = cont.text
            # print("질문 내용 : ",content)
            print()
        except:
            print("질문 내용은 없습니다.")

            # 답변 크롤링

        try:    
            overlays = ".c-heading-answer__content"    
            ans = driver.find_element(By.CSS_SELECTOR, overlays)
            answer = ans.text
            # print("질문 답변 : ",answer)
            print()
            answer_result = answer_result + answer
            print()
        except:
            print("답변 내용은 없습니다.")
    print("최종 추천 : ",answer_result)
    driver.close()    

# 유사도 계산
def similarity(setntence_input):
    ori = []

    result = collection.find({}, {"original_data": 1, "_id": 0}).sort([("$natural", pymongo.DESCENDING)]).skip(1).limit(14) 
    ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    new_setntence = setntence_input
    print("유사도 분석 문장",new_setntence)
    for document in result:
        original_data = document.get("original_data")
        ori.append(original_data)

    sim_result = []

    for i in range(min(14, len(ori))):
        # 이 부분으로 기존 문장들을 벡터화 시킴
        # 코사인 유사도 계산할때는 무조건 벡터화 해야함

        X = ngram_vectorizer.fit_transform([new_setntence, ori[i]])
        similarity = cosine_similarity(X[0], X[1])
        sim_result.append({"new" : new_setntence,"ori" : ori[i],"sim" : similarity[0][0]})

        print()
        print("신규 문장 : ", new_setntence)
        print("기존 문장 : ", ori[i])
        print("두 문장의 유사도 : ", similarity[0][0])
        print() 
        # key = lambda x: x['sim'] sim이라는 키에 키 선택 우선순위가 'sim'이 됨
        # reverse 뒤집다. 원래는 s::-1 이라는 식으로 뒤집에서 정렬하는데 그냥 reverse=True을 사용해서 뒤집은 다음에 정렬함 이게 더 좋은듯?
        sim_result_sorted = sorted(sim_result, key=lambda x: x['sim'], reverse=True)

    if sim_result_sorted:
        most_similar_pair = sim_result_sorted[0]
        print("###################")
        print("가장 유사한 문장")
        print()
        print("신규 문장:", most_similar_pair["new"])
        print()
        print("기존 문장:", most_similar_pair["ori"])
        print()
        print("코사인 유사도:", most_similar_pair["sim"])
        print("###################")
        print()
    else:
        print("유사한 문장이 없습니다.")

def Insert_data():
    user_input = input("유저 번호를 입력하세요:")
    setntence_input = input("문장을 입력하세요: ")
    topic_dic = process_and_store_data(setntence_input, user_input)

    collection.insert_one(topic_dic)
    print("데이터 입력이 완료되었습니다.")
    print()
    similarity(setntence_input)
    Find_Anser(setntence_input)
    
def Delete_data():
    # 데이터 삭제
    Delete_input = input("찾고싶은 유저 번호를 입력하세요:")

    result = collection.delete_one({'_id': Delete_input})  #delete_one pymongo 명령어 콜렉션 내 데이터 삭제임
    if result.deleted_count > 0:
        print("데이터를 성공적으로 삭제했습니다.\n")
    else:
        print("데이터를 삭제하는데 실패했습니다.\n")

def Find_data():
    # 데이터 찾기
    find_input = input("찾고싶은 유저 번호를 입력하세요:")
    query = {'id': find_input}
    result = collection.find(query)

    for document in result:
        print("\n######################")
        print("유저 아이디 : "+document['id'])
        print("입력 문장 : "+document['original_data'])
        print("키워드 추출 : "+str(document['LAD_data']))
        print("######################\n")   

if __name__ == '__main__':
    run()

    while True:
        print("1 : 데이터 입력")
        print("2 : 데이터 찾기")
        print("3 : 데이터 삭제")
        print("4 : 종료\n")
        input_command = input("원하는 기능을 선택하세요: ")

        if input_command == "1":
            Insert_data()
        elif input_command == "2":
            Find_data()
        elif input_command == "3":
            Delete_data()
        elif input_command == "4":
            print("프로그램을 종료합니다.")
            break
        else:
            print("올바른 기능을 선택하세요.")