# 텍스트 자료 수집

import os  # os모듈 import

# (1) 텍스트 디렉터리 경로 지정
print(os.getcwd())   # 기본 작업 디렉터리
txt_data = 'ch08_data/txt_data/'  # 상대 경로 지정

# (2) 텍스트 디렉터리 목록 반환
sub_dir = os.listdir(txt_data)  # txt_data 목록반환
print(sub_dir)  # ['first', 'second']

# (3) 각 디렉터리의 텍스트 자료 수집 함수
def textPro(sub_dir):  # ['first', 'second']
    first_txt = []  # first 디렉터리 텍스트 저장
    second_txt = []  # second 디렉터리 텍스트 저장

    # (3-1) 디렉터리 구성
    for sdir in sub_dir :  # ['first', 'second']
        dirname = txt_data + sdir  # 디렉터리 구성
        file_list = os.listdir(dirname)  # 파일 목록 반환


        # (3-2) 파일 구성
        for fname in file_list:
            file_path = dirname + '/' + fname  # 파일 구성


            # (3-3) file 선택
            if os.path.isfile(file_path):
                try:
                    # (3-4) 텍스트 자료 수집
                    file = open(file_path, 'r')
                    if sdir == 'first':
                        first_txt.append(file.read())
                    else:
                        second_txt.append(file.read())

                except Exception as e:
                    print('예외발생 : ', e)

                finally:
                    file.close()

    return first_txt, second_txt  # 텍스트 자료 반환

# (4) 함수호출
first_txt, second_txt = textPro(sub_dir)    # ['first', 'second']

# (5) 수집한 텍스트 자료 확인
print('first_tex 길이 = ', len(first_txt))  # first_txt 길이 = 10
print('second_tex 길이 = ', len(second_txt))  # second_txt 길이 = 10

# (6) 텍스트 자료 결합
tot_texts = first_txt + second_txt
print('tot_texts의 길이 = ', len(tot_texts))  # tot_texts 길이 = 20

# (7) 전체 텍스트 내용
print(tot_texts)
print(type(tot_texts))





# pickle 저장
# 자료 구조를 갖는 객체를 그대로 파일에 저장하기 위히새 기계어 형식으로 저장한다.
# 이러한 파일을 이진파일(binary file)이라고 함.
# pickle 모듈은 이진파일 형식으로 변수에 저장된 객체를 저장하기 때문에 객체 타입을 그대로 유지하여 파일에 저장하고 읽어올 수 있다.

# (1) pickle 모듈 import
import pickle  # file save

# (2) file save : write binary
pfile_w = open('ch08_data/data/tot_texts.pck', mode='wb')
pickle.dump(tot_texts, pfile_w)

# (3) file load : read binary
pfile_r = open('ch08_data/data/tot_texts.pck', mode='rb')
tot_texts_read = pickle.load(pfile_r)
print('tot_texts의 길이 : ', len(tot_texts_read))
print(type(tot_texts_read))
print(tot_texts_read)





# 이미지 파일 이동
import os  # dif or file path
from glob import glob  # *, ? 파일 검색

# (1) image 파일 경로
print(os.getcwd())
img_path = 'ch08_data/images/'  # 이미지 원본 디렉토리
img_path2 = 'ch08_data/images2/'  # 이미지 이동 디렉토리

# (2) 디렉토리 존재 유무
if os.path.exists(img_path):
    print('해당 디렉토리 존재함')

    # (3) image파일 저장, 파일 이동 디렉토리 생성
    images = []  # 파일저장
    os.mkdir(img_path2)  # 디렉토리 생성

    # (4) images 디렉토리에서 png검색
    for pic_path in glob(img_path + '*.png'):  # png검색

        # (5) 경로와 파일명 분리, 파일명 추가
        img_path = os.path.split(pic_path)
        images.append(img_path[1])  # png 파일명 추가

        # (6) 이진 파일 읽기
        rfile = open(file=pic_path, mode='rb')
        output = rfile.read()

        # (7) 이진파일 쓰기 -> ch08_data/png 폴더 이동
        wfile = open(file=img_path2 + img_path[1], mode='wb')
        wfile.write(output)

    rfile.close(); wfile.close()  # 파일 객체 닫기

else :
    print('해당 디렉터리 없음')


print('png file = ', images)




# 특수 파일 처리
# CSV파일 읽기

# (1) pandas 패키지 import
import pandas as pd
import os
# 현재 작업 디렉토리 확인
print(os.getcwd())

# (2) csv파일 읽기
score = pd.read_csv('ch08_data/data/score.csv')
print(score.info())  # 파일 정보
print(score.head())  # 칼럼명 포함 앞부분 5개 행

# (3) 칼럼 추출
kor = score.kor  # 객체.칼럼명
eng = score['eng']  # 객채['칼럼명']
mat = score['mat']
dept = score['dept']

# (4) 과목별 최고 점수
print('max kor = ', max(kor))
print('max eng = ', max(eng))
print('max mat = ', max(mat))

# (5) 과목별 최하 점수
print('mim kor = ', min(kor))
print('min eng = ', min(eng))
print('min mat = ', min(mat))

# (6) 과목별 평균 점수
from statistics import mean
print('국어점수 평균 : ', round(mean(kor),3))
print('영어점수 평균 : ', round(mean(eng),3))
print('수학점수 평균 : ', round(mean(mat),3))

# (7) dept빈도수
dept_count = {}

for key in dept :
    dept_count[key] = dept_count.get(key, 0) + 1

print(dept_count)


