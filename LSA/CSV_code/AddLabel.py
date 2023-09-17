import csv

# CSV 파일 경로
csv_file_path = "C:/Users/82103/Desktop/Datafile/I/집에서 휴식 일기.csv"

# CSV 파일을 읽어오기 (utf-8 인코딩 사용)
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    data = list(csv_reader)

# 새로운 "label" 열 추가
# 외출0 집콕1
data[0].append("label")  # 첫 번째 행에 "label" 추가
label_value = "1"  # label 열의 값을 1로 설정
for row in data[1:]:  # 두 번째 행부터 마지막 행까지
    row.append(label_value)

# 수정된 데이터를 새로운 파일에 쓰기 (기존 파일을 덮어쓰지 않음)
new_csv_file_path = "C:/Users/82103/Desktop/Datafile/I/집에서 휴식 일기_with_label.csv"
with open(new_csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(data)
