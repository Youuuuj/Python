# 메서드(Overriding) 재정의 : 상속된 메소드를 자식 클래스에서 다시 작성해서 사용할 수 있다.

# (1) 부모 클래스
class Employee :
    nane = None
    pay = 0

    def __init__(self,name):
        self.name = name

    def pay_calc(self):
        pass


class Permanent(Employee):
    def __init__(self, name):
        super().__init__(name)

    def pay_calc(self, base, bonus):
        self.pay = base + bonus

        print('총 수령액 : ', format(self.pay, '3,d'), '원')
class Temporary(Employee):
    def __init__(self,name):
        super().__init__(name)

    def pay_calc(self, tpay, time):
        self.pay = tpay * time
        print('총 수령액 : ', format(self.pay, '3,d'), '원')



p = Permanent('이순신')
p.pay_calc(3000000, 200000)
t = Temporary('홍길동')
t.pay_calc(15000, 80)



# 다형성 : 하나의 참조 변수로 여러 타입의 객체를 참조 할 수 있는 것.
# 부모 객체의 참조 변수로 자식 객체를 다룰 수 있다는 의미.
# 다형성은 클래스의 상속 관계에서만 나올 수 있는 용어.

# (1) 부모클래스
class Flight:

    # 부모 원형 함수
    def fly(self):
        print('날다, fly 원형 메소드')

# (2-1) 자식클래스 : 비행기
class Airplane:

    # 함수 재정의
    def fly(self):
        print('비행기가 날다')


# (2-2) 자식클래스 : 새
class Bird:

    # 함수 재정의
    def fly(self):
        print('새가 날다')


# (2-3) 자식클래스 : 종이비행기
class PaperAirplane:

    # 함수 재정의
    def fly(self):
        print('종이비행기가 날다')


# (3) 객체 생성
# 부모 객체 = 자식 객체(자식1, 자식2)
flight = Flight()
air = Airplane()
bird = Bird()
paper = PaperAirplane()

# (4) 다형성
flight.fly()

flight = air
flight.fly()

flight = bird
flight.fly()

flight = paper
flight.fly()



# 내장클래스

#  import 필요없는  builtin 모듈 내장클래스
# (1) 리스트 열거형 객체 이용
lst = [1,3,5]
for i, c in enumerate(lst):
    print('색인 : ', i, end=", ")
    print('내용 : ', c)

# (2) 딕트 열거형 객체 이용
dict = {'name':'홍길동', 'job':'회사원', 'addr':'서울시'}
for i, k in enumerate(dict):
    print('순서 : ', i, end=', ')
    print('키 : ', k, end=', ')
    print('값 : ', dict[k])



# import 모듈 내장클래스
# (1) 모듈 내장클래스 import
import datetime
from datetime import date, time

# (2) date 클래스
help(date)

today = date(2021, 10, 18)
print(today)

# date 객체 멤버변수 호출
print(today.year)
print(today.month)
print(today.day)

# date 객체 메서드 호출
w = today.weekday()  # Monday==0.....Sunday==6
print('요일 정보 : ', w)

# (3) time 클래스
help(time)

currTime = time(11, 7, 30)
print(currTime)

# time 객체 멤버변수 호출
print(currTime.hour)
print(currTime.minute)
print(currTime.second)

# time 객체 메서드 호출
isoTime = currTime.isoformat()  # HH:MM:SS
print(isoTime)




# 라이브러리 import

# (1) 평균과 제곱근 모듈  import
from statistics import mean
from math import sqrt

# (2) 산술평균
def Avg(data):
    avg = mean(data)
    return avg

# (3) 분산 / 표준편차 함수
def var_sd(data):
    avg = Avg(data)
    diff = [(d-avg)**2 for d in data]
    var = sum(diff) / (len(data)-1)
    sd = sqrt(var)

    return var, sd



# 1. 모듈 추가 (방법 1,2)
# 형식1) import 상위패키지명.하위패키지명. 모듈명
# 형식2) import 상위패키지명.하위패키지명. 모듈명 as 별칭

# 형식1)
import testPackage.sca

data = [1, 3, 1.5, 2, 1, 3.2]

# 산술평균 함수 호출 - 1
print('평균 : ', testPackage.sca.Avg(data))

# 분산과 표준편자 함수 호출 - 1
var, sd = testPackage.sca.var_sd(data)

print('분산 : ', var)
print('표준편차 : ', sd)

# 형식2)
import testPackage.sca as scattering

# 산술평균 함수 호출 - 2
print('평균 : ', scattering.Avg(data))

# 분산과 표준편자 함수 호출 - 2
var, sd = scattering.var_sd(data)

print('분산 : ', var)
print('표준편차 : ', sd)



# 2. 모듈 추가 (방법 3)
# 형식) from 패키지명.모듈명 import 함수명
from testPackage.sca import Avg, var_sd

print("평균 : ", Avg(data))

var, sd = var_sd(data)
print('분산 : ', var)
print('표준편차 : ', sd)



# 시작점 만들기

# 프로그램 시작점 만들기 예
# (1) 평균과 제곱근 모듈  import
from statistics import mean
from math import sqrt

# (2) 산술평균
def Avg(data):
    avg = mean(data)
    return avg

# (3) 분산 / 표준편차 함수
def var_sd(data):
    avg = Avg(data)
    diff = [(d-avg)**2 for d in data]
    # 리스트 내포
    var = sum(diff) / (len(data)-1)
    sd = sqrt(var)

    return var, sd

# 프로그램 시작점
if __name__ == "__main__" :
    data = [1,3,5,7]
    print('평균 = ', Avg(data))
    var, sd = var_sd(data)
    print('분산 = ', var)
    print('표준편차 = ', sd)




# 프로그램 시작업이 없는 경우 예
# (1) 평균과 제곱근 모듈  import
from statistics import mean
from math import sqrt

# (2) 산술평균
def Avg(data):
    avg = mean(data)
    return avg

# (3) 분산 / 표준편차 함수
def var_sd(data):
    avg = Avg(data)
    diff = [(d-avg)**2 for d in data]
    # 리스트 내포
    var = sum(diff) / (len(data)-1)
    sd = sqrt(var)

    return var, sd

# 프로그램 시작점 없음
data = [1,3,5,7]
print('평균 = ', Avg(data))
var, sd = var_sd(data)
print('분산 = ', var)
print('표준편차 = ', sd)


import testPackage.sca
