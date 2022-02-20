# 내장함수

# import  모듈명
# from 모듈명 import 함수명1, 함수명2, ....

# (1) builtins  함수 : 내장함수
help(len)
dataset =list(range(1,6))
print(dataset)

print("len = ", len(dataset))
print("sum = ", sum(dataset))
print("max = ", max(dataset))
print("min = ", min(dataset))

# (2) import  함수 : import 명령어를 이용하여 외부 모듈을 포함시켜야 사용 할 수 있는 함수.
import statistics  # (방법1)
from statistics import variance, stdev  # (방법2)

print('평균 = ', statistics.mean(dataset))
print('중위수 = ', statistics.median(dataset))
print('표본 분산 =  ', variance(dataset))
print('표본 표준편차 = ', stdev(dataset))


# builtins 모듈
# 모듈에서 제공하는 주요 내장함수와 내장 클래스는 dir(x) 내장함수를 이용해서 목록 확인 가능

import builtins
dir(builtins)

# abs(x) : 인수 x를 대상으로 절대값을 반환하는 함수.
abs(-10)

# all(iterable) : 모든 요소가 True 일 때, True를 반환. 영(0)이 아닌 숫자는 True로 해석
all([1, True, 10, -15.2])
all([1, True, 0, -15.2])

# any(iterable) : 하나 이상의 요소가 True 일 때,  True반환. 영(0)을 False로 해석
any([1, False, 10, -15.2])
any([False, 0, 0])

# bin(number) : 10진수 정수를 2진수로 반환한다. 2진수는 '0b'문자열로 시작한다.
bin(7)

# dir(x) : 객체 x에서 제공하는 변수, 내장함수, 내장클래스의 목록을 반환한다.
x = [1, 2, 3, 4, 5]
dir(x)
x.append(6)
x

# eval(expr) : 문자열 수식을 인수로 받아서 계산 가능한 파이썬 수식으로 변환.
eval("10 + 20")
eval(10 + "20 + 30")  # 오류 발생
eval("20 + 30") + 10

# hex(number) : 10진수 정수를 16진수로 반환. 16진수는 '0x'문자열로 시작
hex(10)

# oct(number) : 10진수 정수를 8진수로 반환. 8진수는 '0o'문자열로 시작
oct(10)

# ord(charactor) : charactor값을 아스키 값으로 반환한다. 숫자 0은 48, 영문자 대문자 A는 65, 영문자 소문자 a는 97이다.
ord('0')
ord('9')

# pow(x,y) : x에 대한 y의 제곱을 계산하여 반환
pow(2, 3)

# round(number) : 실수를 인수로 하여 반올림을 수행하는 결과를 반환한다.
round(3.14159)
round(3.14159, 3)  # ,x 는 소수점 자리

# sorted(iterabla) : 반복 가능한 원소들을 대상으로 오름차순 또는 내림차순 정렬한 결과를 반환
sorted([1, 8, 3, 5, 2])
sorted([1, 8, 3, 5, 2], reverse=True)  # reverse=True는 내림차순

# zip(iterable*) : 반복 가능한 객체와 객체 간의 원소들을 묶어서 튜플로 반환.
# zip()함수에서 반환된 결과를 확인하기 위해서 list 클래스를 이용하여 리스트 자료구조로 변환해야한다
zip([1, 3, 5], [2, 4, 6])
list(zip([1, 3, 5], [2, 4, 6]))



