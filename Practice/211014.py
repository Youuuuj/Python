# 사용자 정의 함수
#def 함수명(매개변수) :
#    실행문
#    실행문
#    return 값

# (1) 인수가 없는 경우
import random


def userFunc1() :
    print('인수가 없는 함수')
    print('userFunc1')

userFunc1()

# (2) 인수가 있는 함수
def userFunc2(x, y) :
    print('userFunc2')
    z = x + y
    print('z = ', z)

userFunc2(20, 10)

# (3) return 있는 함수
def userFunc3(x, y) :
    print('userFunc3')
    tot = x + y
    sub = x - y
    mul = x * y
    div = x / y

    return tot, sub, mul, div

userFunc3(20,10)

# 실인수 : 키보드 입력
x = int(input('x 입력 : '))
y = int(input('y 입력 : '))

a, b, c, d = userFunc3(x, y)  # 앞의 4글자는 리턴값이 4개라서 아무거나 4개 써도 무관함
print('tot = ', a)
print('sub = ', b)
print('mul = ', c)
print('div = ', d)


# 산포도 구하기 : 평균으로부터 얼마나 값이 분산되어 있는지의 정도를 나타내는 척도로 분산과 표준편차를 사용한다.
from statistics import mean, variance
from math import sqrt

dataset = [2, 4, 5, 6, 1, 8]

# (1) 산술평균
def Avg(data) :
    avg = mean(data)
    return avg

print('산술평균 = ', Avg(dataset))

# (2) 분산/표준편차

def var_sd(data):
    avg = Avg(data)  # 함수호출
    diff = [(d - avg)**2 for d in data]

    var = sum(diff) / (len(data)-1)
    sd = sqrt(var)

    return var, sd

# (3) 함수호출
v, s = var_sd(dataset)
print('분산 = ', v)
print('표준편차 = ', s)



# 피타고라스 정리
def pytha(s, t):
    a = s**2 - t**2
    b = 2 * s * t
    c = s**2 + t**2
    print('3변의 길이 : ',a, b, c)

pytha(2, 1)  # s, t의 인수는 양의 정수를 갖는다



# 몬테카를로 시뮬레이션 : 현실적으로 불가능한 문제의 해답을 얻기 위해서
# 난수의 확률 분포를 이용해 모의실험으로 근사적 해를 구하는 기법

# 단계 1 : 동전 앞면과 뒷면이 난수 확률 분포 함수 정의
import random


def coin(n):
    result = []
    for i in range(n):
        r = random.randint(0, 1)
        if (r == 1):
            result.append(1)  # 앞면
        else:
            result.append(0)  # 뒷면
    return result

print(coin(10))


# 단계 2 : 몬테카를로 시뮬레이션 함수 정의
def montaCoin(n):
    cnt = 0
    for i in range(n):
        cnt += coin(1)[0]  # coin 함수 호출

    result = cnt / n  # 누적 결과를 시행 횟수(n)로 나눈다.
    return result




# 단계 3 : 몬테카를로 시뮬레이션 함수 호출
print(montaCoin(10))
print(montaCoin(30))
print(montaCoin(100))
print(montaCoin(1000))
print(montaCoin(10000))


# 중심 극한 정리 : 표본의 크기가 커질수록 근사적으로
#                표본의 평균이 모평균과 같고, 분산이 모분산과 같은 정규분포를 취한다는 이론



# 가변인수 함수 : 하나의 매개변수로 여러 개의 실인수를 받을 수 있는 가변인수.
#               실인수를 받는 매개변수 앞부분에 * 또는 ** 기호를 붙인다.
#               *매개변수는 튜플 자료구조로,  **매개변수는 딕트 자료구조로 받는다.

# (1) 튜플형 가변인수
def Func1 (name, *names) :
    print(name)  # 실인수 : 홍길동
    print(names)  # 실인수 : ('이순신', '유관순')

Func1("홍길동", "이순신", '유관순')

# statistic 모듈 import
from statistics import mean, variance, stdev

# (2) 통계량 구하는 함수 :
def statis(func, *data) :
    if func == 'avg':
        return mean(data)
    elif func == 'var':
        return variance(data)
    elif func == 'std':
        return stdev(data)
    else :
        return 'TypeError'

# statis 함수 호출

print('avg = ', statis('avg', 1,2,3,4,5))
print('var = ', statis('var', 1,2,3,4,5))
print('std = ', statis('std', 1,2,3,4,5))

# (3) 딕트형 가변인수
def emp_func(name, age, **other):
    print(name)
    print(age)
    print(other)

# emp_func 호출
emp_func('홍길동', 35, addr='서울시', height=175, weight=65)



# 람다함수 : 정의와 호출을 한번에 하는 익명함수.
#           복잡한 함수호출 과정을 생략해서 처리시간을 단축되고, 코드의 가독성을 제공하는 이점 가짐
#           별도의 함수이름, 인수를 나타는 괄호, return 명령어 없음
# lambda 매개변수 : 실행문(반환값)

# (1) 일반 함수
def Adder(x,y):
    add = x+y
    return add

print('add = ', Adder(10,20))

# (2) 람다 함수
print('add = ', (lambda x,y : x+y)(10,20))


# 스코프(Scope) : 특정지역에서만 사용되는 지역변수와 지역에 상관없이 전 지역에서 사용도는 전역변수로 분류됨.
#                이처럼 변수가 사용되는 범위를 스코프라고함
# def 함수명(인수) : global 전역변수

# (1) 지역변수
x = 50  # 전역변수
def local_func(x):
    x += 50  # 지역변수 -> 종료 시점 소멸
local_func(x)
print('x = ', x)

# (2) 전역변수
def global_func():
    global x  # 전역변수 x 사용
    x += 50  # x = x + 50

global_func()
print('x=', x)




# 중첩함수 : 함수 냅에 또 다른 함수가 내장된 형태를 의미.

# 일급함수 : 중첩함수는 외부함수나 내부함수를 변수에 저장할 수 있는데, 이러한 특성을 갖는 함수를 일급함수라고 함.
# 함수클로저 : 외부함수가 종료되어도 내부함수에서 선언된 변수가 메모리에 남아 있다면 내부함수 활용 가능.

# (1) 일급함수
def a():  # outer
    print('a 함수')
    def b():  # inner
        print('b 함수')
    return b
b = a()
b()


# (2) 함수클로저
data = list(range(1,101))
def outer_func(data):
    dataSet = data  # 값(1~100) 생성
    # inner
    def tot():
        tot_val = sum(dataSet)
        return tot_val
    def avg(tot_val):
        avg_val = tot_val / len(dataSet)
        return avg_val
    return tot, avg  # inner 반환

# 외부 함수 호출 : data 생성
tot, avg = outer_func(data)

# 내부 함수 호출
tot_val = tot()
print('tot = ', tot_val)
avg_val = avg(tot_val)
print('avg = ', avg(tot_val))



# 중첩함수 역할
from statistics import mean
from math import sqrt

data = [4, 5, 3.5, 2.5, 6.3, 5.5]

# (1) 외부함수 : 산포도함수
def scattering_func(data):  #outer
    dataSet = data  # data 생성

    # (2) 내부함수 : 산술평균 반환
    def avg_func():
        avg_val = mean(dataSet)
        return avg_val

    # (3) 내부함수 : 분산 반환
    def var_func(avg):
        diff = [(data-avg)**2 for data in dataSet]
        var_val = sum(diff) / (len(dataSet)-1)
        # print(sum(diff)  차의 합
        return var_val

    # (4) 내부함수 : 표준편차 반환
    def std_func(var):
        std_val = sqrt(var)
        return std_val

    # 함수 클로저 반환
    return avg_func, var_func, std_func   # 리턴 값이 3개.

# (5) 외부함수 호출
avg, var, std = scattering_func(data)   # avg, var, std <- 아무런 글자 3개 써도 리턴값이 차례대로 지정되어짐.

# (6) 내부 함수 호출
print('평균 : ', avg())
print('분산 : ', var(avg()))
print('표준편차 : ', std(var(avg())))


# 획득자, 지정자, nonlocal
# 획득자 함수(getter) : 함수 내부에서 생선한 자료를 외부로 반환하는 함수로 반드시 return 명령문을 갖는다.
# 지정자 함수(setter) : 함수 내부에서 생선한 자료를 외부에서 수정하는 함수로 반드시 매개변수 갖는다.
# 만약 외부함수에서 생선된 자료를 수정할 경우에는 해당 변수에 nonlocal 명령어를 쓴다.
# def 외부함수 () :
#       변수명 = 값
#     def 내부함수 () :
#           nonlocal 변수명


# (1) 중첩함수 정의
def main_func(num):
    num_val = num  # 자료생성
    def getter():  # 획득자 함수, 리턴 있음
        return num_val
    def setter(value):  # 지정자 함수, 인수 있음
        nonlocal num_val
        num_val = value

    return getter, setter

# (2) 외부함수 호출
getter, setter = main_func(100)

# (3) 획득자 호출
print('num : ', getter())

# (4) 지정자 획득
setter(200)
print('num : ', getter())



# 함수 장식자 : 기존 함수의 시작 부분과 종료 부분에 기능을 장식해서 추가해 주는 별도의 함수
# (1) 래퍼 함수
def Trash(func):
    def decorated() :
        func()
        print('이 양심없는')
        print('인간들아')
    return decorated

# (2) 함수 장식자 적용
@Trash
def trash() :
    print('제대로 쓰레기통에')

# (3) 함수 호출
trash()




# 재귀함수 : 함수 내부에서 자신의 함수를 반복적으로 호출하는 함수를 의미한다.
#           재귀함수는 반복적으로 함수를 호출하기 때문에 반드시 함수내에서 반복을 탈출하는 조건이 필수다.

# (1) 재귀함수 정의 : 1~n 카운트
def Counter(n) :
    if n == 0 :
        return 0  # 종료 조건
    else :
       # 재귀 호출
        Counter(n-1)
        print(n, end=" ")




# (2) 함수 호출 1
print('n = 0 : ', Counter(0))

# (3) 함수 호출 2
Counter(5)


# 누적합

# (1) 재귀함수 정의 : 1~n 누적합
def Adder(n):
    if n == 1 :
        return 1
    else :
        result = n + Adder(n-1)

        print(n, end = ' ')
        return result

# (2) 함수 호출 1
print('n = 1 : ', Adder(1))

# (3) 함수 호출 2
print(('n = 5 : ', Adder(5))

Adder(5)

if height == 1:
    print('*')
else:
    StarCount(height - 1)
print('*' * height, end='\n')


