import pandas as pd

# 첫 번째 CSV 파일을 먼저 읽습니다.
first_file = pd.read_csv('집에서 휴식 일기.csv', encoding='utf-8')

# 두 번째부터 마지막 CSV 파일까지를 순회하며 읽고 첫 번째 행을 제외합니다.
combined_data = first_file.copy()  # 첫 번째 파일의 내용을 복사합니다.

# 나머지 파일들을 합칩니다.
for file_name in ['집콕 일기.csv']:  # 필요한 파일 이름을 열거합니다.
    df = pd.read_csv(file_name, encoding='utf-8')
    df = df.iloc[1:]  # 첫 번째 행을 제외합니다.
    combined_data = pd.concat([combined_data, df])

# 합친 데이터를 새로운 CSV 파일로 저장합니다.
combined_data.to_csv('I_combined.csv', index=False, encoding='utf-8')
