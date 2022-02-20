for i in range(10, 20):
    print('--- {}단 ---'.format(i))

    for j in range(1, 10):
        print('%d * %d = %d' % (i, j, i*j))


string = """나는 홍길동입니다.
주소는 서울시 입니다.
나이는 35세 입니다."""

sents = []  # 문장 저장
words = []  # 단어 저장

for sen in string.split(sep="\n"):
    sents.append(sen)

    for word in sen.split():
        words.append(word)

print('문장 : ', sents)
print('문장수 : ', len(sents))
print('단어 : ', words)
print('단어수 : ', len(words))

str_var = str(object='string')
print(str_var)
print(type(str_var))
print(str_var[0])
print(str_var[-1])


lst = [1, 2, 3, 4, 5]
print(lst)
print(type(lst))

for i in lst:
    print(lst[:i])  # 파이썬에서 시작은 0부터


x = list(range(1, 11))
print(x)
print(x[:5])
print(x[-5:])
print('index 2씩 증가 ')
print(x[::2])
print(x[1::2])

a = ['a', 'b', 'c']
print(a)

b = [10, 20, a, 5, True, '문자열']
print(b)

print(b[0])
print(b[2])
print(b[3])

# 추가, 삭제, 수정, 삽입
# (1) 단일 리스트 객체 생성
num = ['one', 'two', 'three', 'four']
print(num)
print(len(num))

# (2) 리스트 원소 추가
num.append('five')
print(num)

# (3) 리스트 원소 삭제
num.remove('five')
print(num)

# (4) 리스트 원소 수정
num[3] = 4
print(num)

# (5) 리스트 원소 삽입
num.insert(0, 'zero')
print(num)

num.insert(4, 'four')
print(num)


# 리스트 연산
# (1) 리스트 결합
x = [1, 2, 3, 4]
y = [1.5, 2.5]
z = x + y
print(z)

# (2) 리스트 확장
x.extend(y)
print(x)

# (3) 리스트 추가(중첩 리스트가 된다)
x.append(y)
print(x)

# (4) 리스트 두배 확장
lst = [1, 2, 3, 4]  # list 생성
result = lst * 2  # 각 원소 연산 안됨
print(result)


# 리스트 정렬과 요소 검사 : 리스트 원소 중에서 특정 갑의 존재 유무를 알려주는 기능 제공., 값있으면 True, 없으면  False 반환

# (1) 리스트 정렬
print(result)
result.sort()  # 오름차순 정렬
print(result)
result.sort(reverse=True)  # 내림차순 정령
print(result)

# (2) 리스트 요소 검사
import random
r = []  # 빈 list
for i in range(5):
    r.append(random.randint(1, 5))

print(r)

if 4 in r:
    print("있음")
else:
    print("없음")

# scala 변수 : 한 개의 값을 갖는 변수로 값의 크기를 갖는다. ex) x = 10
# vector 변수 : 여러 개의 값을 갖는 변수로 값의 크기와 뱡향을 갖는다. ex) x = [10,20,30,40,50]



# 리스트내포 : list안에서 for와 if를 사용하는 문법.
# 형식1 : 변수 = [실행문 for 변수 in 열거형객체]
# 형식2 : 변수 = [실행문 for 변수 in 열거형객체 if조건식]
# 형식3 : 변수 = [값1(True) if 조건문 else 값2(False) for 변수 in 열거형 객체]


# 형식1) 변수 = [ 실행문 for ~ ]
x = [2, 4, 1, 5, 7]
# print(x**2) 에러남 왜냐 x는 리스트이기 때문에

lst = [i**2 for i in x]  # x변량에 제곱 계산
print(lst)

# 형식2) 변수 = [ 실행문 for ~ if ~ ]
# 1~10 -> 2의 배수 추출 -> i*2 -> list저장
num = list(range(1, 11))
print(num)
lst2 = [i*2 for i in num if i % 2 == 0]
print(lst2)


# 튜플 :  순차 자료구조. 리스트와 비슷. 읽기전용이므로 원소 수정 삭제 안돼. 리스트 비해서 처리속도 빠름.
# 특징 : 순서 자료를 갖는 열거형객체를 생성, (,)를 사용하여 순서대로 값을 나열.
#       값의 자료형은 숫자형, 문자형, 논리형 등을 함께 사용할 수 있다.
#       색인(index)사용가능. 슬라이싱,연결,반복,요소검사 가능
#       읽기 전용이기 때문에, 값을 추가, 삽입, 수정, 삭제 불가능
#       리스트보다 처리속도 빠름

# (1) 원소가 한개인 경우
t = (10,)
print(type(t))

# (2) 원소가 여러 개인 경우
t2 = (1, 2, 3, 4, 5, 3)
print(t2)
print(type(t2))

# 두개 이상의 데이터를 콤마로 연결하면 괄호상관없이 튜플!
t3 = 1, 2, 3, 4
print(t3)
print(type(t3))

# (3) 튜플 색인
print(t2[0], t2[1:4], t2[-1])

# (4) 수정 불가
t2[0] = 10  # error

# (5) 요소 반복
for i in t2:
    print(i, end=" ")  # end옵션은 한줄에 붙여서 쓰기

# (6) 요소 검사
if 6 in t2:
    print("6 있음")
else:
    print("6 없음")


t4 = 1, 9, 4, 3
print(t4)
t5 = (t4[0], t4[3], t4[2], t4[1])
print(t5)
t6 = t4[1], t4[2]
t7 = t4[1:3]
print(t6, t7)

lst2 = list(t4)
lst2.sort()
t9 = tuple(lst2)
print(type(t9))

# 튜플 자료형 변환
lst = list(range(1, 6))
t8 = tuple(lst)
print(t8)

# 튜플 관련 함수
print(len(t8), type(t3))  # 튜플 내의 변수의 갯수, 형식
print(t8.count(3))  # 튜플 내의 count(x) x의 개수
print(t8.index(4))  # 인덱스 4번째


# 비순서 자료구조 : 칸막이로 구분되지 않고, 공토의 영역에 값들이 적재된다.
# set, dict 사용

# set : 여러 개의 자료를 비 순서로 적재하는 가변 길이 비순차 자료구조를 생성하는 함수
# 공통의 영역에 값들이 적재
# 특징 : 비순서 자료구조 갖는 열거형 객체. 중괄호{}안에 콤마,를 이용 원소를 구분
#       중복 허용x. 순서x 색인(index)사용 불가. 객체에서 제공하는 함수를 이용하여 추가, 삭제 및 집합 연산 가능

# (1) 중복 불가
s = {1, 3, 5, 3, 1}
print(len(s))
print(s)

# (2) 요소 반복
for d in s:
    print(d, end=' ')

# (3) 집합 관련 함수
s2 = {3, 6}
print(s.union(s2))  # 합집합
print(s.difference(s2))  # 차집합
print(s.intersection(s2))  # 교집합

# (4) 추가, 삭제 함수
s3 = {1, 3, 5}
print(s3)

s3.add(7)  # 원소 추가
print(s3)

s3.discard(3)  # 원소 삭제
print(s3)

# 중복 제거 : 중복을 허용하지;않는 셋의 특징을 이용해 리스트의 중복 원소를 제거하는데 이용

# 중복원소를 갖는 리스트
gender = {'남', '여', '남', '여'}
# 중복 원소 제거
sgender = set(gender)  # list -> set
print(sgender)
print(type(sgender))
lgender = list(sgender)  # set -> list
print(lgender)

print(lgender[0])



# 딕트(dict) : 사전형으로 여러 개의 자료를 비 순서로 적재하는 가변 길이 비순차 자료구조.
#             키(key)에 값(value)을 저장해 키를 통해서 값을 참조하는 형식
# 특징 : 사전 형식으로 비순서 자료구조를 갖는 열거형객체를 생성할 수 있음
#       {'키' : '값'}의 쌍으로 원소를 입력하고, 콤마(,)를 이용하여 원소 구분
#       '키'는 중복이 허용되지 않고, '값'은 중복 허용, '키'에는 list형식 불가, 튜플은 가능
#       색인(index) 대신에 키(key)를 이용해서 '값'을 참조
#       키를 색인으로 이용할 수 있기 때문에 원소 수정, 삭제, 추가 등이 가능
# 형식 : 변수 = {'키' : '값', '키' : '값', '키' : '값', .... '키' : '값'}

# (1) dict 생성 방법1
dic = dict(key1=100, key2=200, key3=300)
print(dic)

# (2) dict 생성 방법2
person = {'name': '홍길동', 'age': 35, 'address': '서울시'}
print(person)
print(person['name'])
print(type(dic), type(person))
