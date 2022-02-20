# 유준


# 1번
import os  # os모듈 import
os.getcwd()  # 현재 경로확인


file = open('ch08_data/data/ftest.txt', mode='r')

lines = file.readlines() # 줄 단위 전체 읽기
docs = []  # 문장 저장할 리스트
words = []  # 단어 저장할 리스트

# (1) 문장에서 \n 제거
for line in lines :
    docs.append(line.strip())

    # (2) 문장 -> 단어
    for word in line.split():
        words.append(word)

print('문장내용')
print(docs)
print('문장수 : ', len(docs))
print('단어내용')
print(words)
print('단어수 : ', len(words))





# 2번
import os # os import
import pandas as pd  # pandas패키지 import
from statistics import mean  # 평균 구하기 위해 mean import

# 1. file read
print(os.getcwd())  # 현재 디렉토리 확인
emp = pd.read_csv('ch08_data/data/emp.csv', encoding='utf-8')
print(emp.info())  # 파일 정보
print(emp.head())  # 칼럼명 포함 앞부분 5개의 행 확인

pay = emp.Pay
name = emp.Name
print(pay)
print(name)


pay_list = pay.values.tolist()  # values.tolist() 메소드를 이용해 리스트로 변환
name_list = name.values.tolist()
print(type(pay_list),type(name_list))
print('pay : ', pay_list)
print('name : ', name_list)



print('관측치의 길이 : ',len(emp))
print('전체 급여 평균 : ', float(round(mean(pay))))  # float()함수를  이용해 실수로 변경

for i in range(len(pay_list)):
    if pay_list[i] == min(pay_list):
        print('최저급여 : %d, 이름 : %s' %(min(pay_list),name_list[i]))
    elif pay_list[i] == max(pay_list):
        print('최고급여 : %d, 이름 : %s' %(max(pay_list),name_list[i]))
    else:
        pass


# 하고나서 보니까 리스트변경 필요없었음
for i in range(len(pay)):
    if pay[i] == min(pay):
        print('최저급여 : %d, 이름 : %s' %(min(pay),name[i]))
    elif pay[i] == max(pay):
        print('최고급여 : %d, 이름 : %s' %(max(pay),name[i]))
    else:
        pass

