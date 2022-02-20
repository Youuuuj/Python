# 유준

# 1
class Rectangle :
    width = 0
    height = 0

    def __init__(self,width, height):
        self.width = width
        self.height = height

    def area_calc(self):
        area = self.width * self.height
        print('사각형의 넓이 : %d'%area)

    def circum_calc(self):
        circum = (self.width + self.height) * 2
        print('사각형의 둘레 : %d'%circum)
        print('-' * 40)


print('사각형의 넓이와 둘레를 계산합니다.')
w = int(input('사각형의 가로 입력 : '))
h = int(input('사각형의 세로 입력 : '))
print('-' * 40)
rec = Rectangle(w,h)
rec.area_calc()
rec.circum_calc()



# 2
from statistics import mean
from math import sqrt

x = [5, 9, 1, 7, 4, 6]

# 산포도 클래스
class Scattering:
    data = []

    def __init__(self,data):
        self.data = data

    def var_func(self):
        avg = mean(self.data)
        diff = [(d - avg) ** 2 for d in self.data]
        self.var = sum(diff) / (len(self.data)-1)
        return self.var

    def std_func(self):
        self.sqrt = sqrt(self.var)
        return self.sqrt



Result = Scattering(x)


print('분산 : ', Result.var_func())
print('표준편차 : ', Result.std_func())

# 3
class Person:
    name = None
    gender = None
    age = 0

    # 생성자
    def __init__(self,name,gender,age):
        self.name = name
        self.gender = gender
        self.age = age

    # 메서드
    def disply(self):
        print(('이름 : {}, 성별 : {} \n나이 : {}').format(self.name, self.gender, self.age))
        print('=' * 30)


name = input('이름 : ')
age = int(input('나이 : '))
gender = input('성별(male/female) : ')
print('=' * 30)

# 객체 생성
p = Person(name, gender, age)
p.disply()



# 4

# 부모 클래스
class Employee:
    name = None
    pay = 0

    def __init__(self,name) :
        self.name = name

# 자식 클래스 : 정규직
class Permanent(Employee):
    def __init__(self,name):
        super().__init__(name)

    def pay_calc(self, base, bonus):
        self.pay = base + bonus
        print('이름 : ', self.name)
        print('급여 : ', format(self.pay,'3,d'))

# 자식 클래스 : 임시직
class Temporary(Employee):
    def __init__(self,name):
        super().__init__(name)

    def pay_calc(self, tpay, time):
        self.pay = tpay * time
        print('이름 : ', self.name)
        print('급여 : ', format(self.pay, '3,d'))



def EMP(x):
    if empType == 'P' or empType == 'p':
        name = (input('이름 : '))
        base = int(input('기본급 : '))
        bonus = int(input('상여금 : '))

        print('=' * 30)
        print('고용형태 : 정규직')
        P = Permanent(name)
        P.pay_calc(base,bonus)

    elif empType == 'T' or empType == 't':
        name = (input('이름 : '))
        time = int(input('작업시간 : '))
        tpay = int(input('시급 : '))

        print('=' * 30)
        print('고용형태 : 임시직')
        T = Temporary(name)
        T.pay_calc(tpay,time)

    else :
        print('='*30)
        print('입력 오류')

empType = input('고용형태 선택(정규직<P>, 임시직<T>) : ')
EMP(empType)



# 5

import myCalcPackage.calcModule as examply

x = 10
y = 5
print('x = %d; y = %d 일 때'%(x,y))
print("Add= ", examply.Add(x,y))
print("Sub= ", examply.Sub(x,y))
print("Mul= ", examply.Mul(x,y))
print("Div= ", examply.Div(x,y))


