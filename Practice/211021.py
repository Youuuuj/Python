# 예외처리

# 형식
# try :
#   예외발생코드
# except 예외처리 클래스  as 변수 :
#   예외처리 코드
# finally :
#   항상 실행 코드


# 간단한 예외처리
# (1) 예외 발생 코드
print('프로그램 시작 !!!')
x= [10, 20, 25.2, 'num', 14, 51]

for i in x :
    print(i)
    y = 1**2  # 예외발생
    print('y = ', y)


print('프로그램 종료')


# (2)
print('프로그램 시작 !!!')

for i in x :
    try:
        y = i**2
        print('i = ', i, ', y = ', y)
    except :
        print('숫자아님 : ', i)

print('프로그램 종료')



# 다중 예외처리
# 형식 : except 예외처리 클래스 as 변수 :

# 중첩 예외처리 예
# 유형별 예외처리
print('\n유형별 예외처리')
try:
    div = 1000 / 2.53
    print('div = %5.2f' %(div))  # 정상
    div = 1000 / 0  # 1차 : 산술적 예외
    f = open('C:\\test.txt')  # 2차 : 파일열길
    num = int(input('숫자입력 : '))  # 3차 : 기타예외
    print('num = ', num)


# 다중 예외처리 클래스
except ZeroDivisionError as e:  # 산술적 예외처리
    print('오류정보  : ', e)
except FileNotFoundError as e:  # 파일 열기 예외처리
    print('오류정보 : ', e)
except Exception as e:  # 기타 예외처리
    print('오류정보 : ', e)


finally:
    print('finally 영역 - 항상 실행되는 영역')



# 텍스트 파일
# 텍스트 파일 입출력  : 텍스트 파일을 읽고 쓰기 위해서 io모듈에서 open()함수를 이용.
# 형식 : opne(file, mode, encoding)

# file : 파일의 경로와 파일명을 지정
# mode : 읽기모드, 쓰기모드, 쓰기+추가 모드 등을 정해진 문자(character)로 지정.
#        'r' : 읽기용 파일 객체를 연다(기본값)
#        'w' : 쓰기용 파일 개체를 연다
#        'x' : 쓰기용 파일을 새로 만들어서 연다.(기존파일 있으면  Error)
#        'a' : 기존 파일의 맨 마지막에 추가용 파일 객체를 연다.
#        'b' : 이진파일(binary file)형식으로 읽기/쓰기 파일 객체를 연다.
# encoding : 인코딩 또는 디코딩에 사용되는 인코딩의 이름을 지정하는 속성으로 텍스트 모드에서만 사용.
#            기본 인코딩은 플랫폼에 따라 다르지만 파이썬에서 지원하는 인코딩 목록은 codec 모듈을 참조.

# (1) 현재 작업디렉터리
import os
print('\n현재 경로 : ', os.getcwd())

# (2) 예외처리
try:
    # (3) 파일읽기
    ftest1 = open('ch08_data/data/ftest.txt', mode = 'r')
    print(ftest1.read())  # 파일 전체 읽기

    # (4) 파일쓰기
    ftest2 = open('ch08_data/data/ftest2.txt', mode = 'w')
    ftest2.write('my first text ~~~')  # 파일 쓰기

    # (5) 파일 쓰기 + 내용추가
    ftest3 = open('ch08_data/data/ftest2.txt', mode = 'a')
    ftest3.write('\nmy second text ~~~')  # 파일 쓰기(추가)

except Exception as e:
    print('Error 발생 : ', e)

finally:
    ftest1.close()
    ftest2.close()
    ftest3.close()



# 텍스트 자료 읽기
# read() : 전체 텍스트 자료를 한 번에 읽어온다. 읽어온 자료는 문자열(str) 자료형으로 반환된다.
# readlines() : 전체 텍스트 자료를 줄 단위로 읽어온다. 읽어온 자료는 리스트(list)자료형으로 반환
# readline() : 한 줄 단위로 읽어온다. 읽어온 자료는 문자열(str) 자료형으로 반환


# 텍스트 자료 읽기 예
# 파일 읽기 관련 함수
try :

    # (1) read() : 전체 텍스트 읽기
    ftest = open('ch08_data/data/ftest.txt', mode='r')
    full_text = ftest.read()
    print(full_text)
    print(type(full_text))

    # (2) readlines() : 전체 텍스트 줄 단위 읽기
    ftest = open('ch08_data/data/ftest.txt', mode='r')
    lines = ftest.readlines()  # 리스트 반환
    print(lines)
    print(type(lines))
    print('문단수 : ', len(lines))

    # (3) list -> 문장 추출
    docs = []  # 문장 저장
    for line in lines:
        print(line.strip())
        docs.append(line.strip())

    print(docs)

    # (4) readline : 한 줄 읽기
    ftest = open('ch08_data/data/ftest.txt', mode='r')
    line = ftest.readline()
    print(line)
    print(type(line))

except Exception as e:
    print('Errop 발생 : ', e)

finally:
    # 파일 객체 닫기
    ftest.close()




# with 블록과 인코딩 방식
# 형식 : with open(file, mode, encoding) as 참조변수 :
#           pass
# with블록을 이용하면 블록 내에서 참조변수를 이용하여 파일 객체를 사용할 수 있고, 블록을 벗어나면 자동으로 객체가 close된다.

try:
    with open('ch08_data/data/ftest3.txt', mode='w', encoding='utf-8') as ftest :
        ftest.write('파이썬 파일 작성 연습')
        ftest.write('\n파이썬 파일 작성 연습2')
        # with 블록 벗어나면 자동 close

    with open('ch08_data/data/ftest3.txt', mode='r', encoding='utf-8') as ftest :
        print(ftest.read())

except Exception as e:
    print('Error : ', e)
finally:
    pass


# 파일시스템
# os모듈의 파일과 디렉터리 관련 함수 예
import os  # os 모듈 import

# 현재 작업 디렉터리 경로 확인
os.getcwd()

# 작업 디렉터리 변경
os.chdir('ch08_data')
os.getcwd()

# 현재 작업 디렉터리 목록 : list반환
os.listdir('.')

# 디렉터리 생성
os.mkdir('test')
os.listdir('.')

# 디렉터리 이동 : 'test'이동
os.chdir('test')
os.getcwd()

# 여러 디렉터리 생성 : 'test2', 'test3' 생성
os.makedirs('test2/test3')
os.listdir('.')

# 디렉터리 이동
os.chdir('test2')
os.listdir('.')

# 디렉터리 삭제
os.rmdir('test3')
os.listdir('.')

# 상위 디렉터리 이동 : 상위 디렉터리 2개 이동
os.chdir('../..')
os.getcwd()

# 여러 개의 디렉터리 삭제 : 'test', 'test2'삭제
os.removedirs('test/test2')
os.getcwd()



# os.path 모둘의 경로 관련 함수
# 현재 경로 확인
os.getcwd()

# 경로 변경
os.chdir('ch08_data')
os.getcwd()

# lecture 디렉터리의 step01_try_except.py 파일 절대경로
os.path.abspath('lecture/step01_try_except.py')

# step01_try_except.py 파일의 디렉터리 이름
os.path.dirname('lecture/step01_try_except.py')

# workspace 디렉터리 유무 확인
os.path.exists('C:\\Users\\You\\Desktop\\빅데이터 수업자료\\파이썬 연습')

# step01_try_except.py 파일 유무 확인
os.path.isfile('lecture/step01_try_except.py')

# lecture 디렉터리 유무 확인
os.path.isdir('lecture')

# 디렉터리와 파일 분리
os.path.split('c\\test\\test1.txt')

# 디렉터리 파일 결합
os.path.join('c\\test','test1.txt')

# step01_try_except.py 파일 크기
os.path.getsize('lecture/step01_try_except.py')