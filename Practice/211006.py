import keyword

var1 = "Hello python"
print (var1)
print(id(var1))


var1 = 100
print (var1)
print(id(var1))

#예약어 확인
import keyword #모듈 임포트

python_keyword = keyword.kwlist
print(python_keyword)


var1 = "Hello python"
print (var1)
print(type(var1))

var1 = 100
print (var1)
print(type(var1))

var2 = 150.00
print(var2)
print(type(var2))

var3 = True
print(var3)
print(type(var3))


# 실수 -> 정수
a = int(10.5)
b = int(20.42)
add = a + b
print ('add = ', add)

# 정수 -> 실수

a = float(10)
b = float(20)
add2 = a + b
print("add2 = ", add2)

#논리형 -> 정수
print(int(True)) # 1
print(int(False)) # 0

#문자형 -> 정수

st = '10'
print(st*2)

num1 = 100
num2 = 20

add = num1 + num2 #덧셈
print('add = ', add)

sub = num1 - num2 #
print(sub)

mul = num1 * num2
print(mul)

div = num1 / num2
print(div)

div2 = num1 % num2
print(div2)

square = num1**2
print(square)

# (1) 동등비교
bool_result = num1 == num2 # 두 변수의 값이 같은지 비교
print(bool_result)

bool_result = num1 != num2 # 두 변수의 값이 다른지 비교
print(bool_result)

# (2)크기 비교

bool_result = num1 > num2
print(bool_result)

bool_result = num1 >= num2
print(bool_result)

bool_result = num1 < num2
print(bool_result)

bool_result = num1 <= num2
print(bool_result)


# 두 관계식 중 하나라도 같은지 판단
log_result = num1 >= 50 and num2 <= 10
print(log_result)

log_result = num1 >= 50 or num2 <= 10
print(log_result)
log_result = num1 >= 50
print(log_result)

log_result = not(num1 >= 50)
print(log_result)

i = tot = 10
i += 1 # i = i + 1
tot += i
print(i, tot)

print('출력', end=' , ')
print('출력2')

print('출력 , 출력2')


v1, v2 = 100, 200
print(v1, v2)

v2, v1 = v1, v2
print(v1, v2)

#패킹(packing)할당
lst = [1, 2, 3, 4, 5]
v1, v2, v3 = lst
print(v1,v2, v3)



# (1) 문자형 숫자 입력
num = input("숫자입력 : ")
print('num type : ', type(num)) # <class 'str'>

print('num = ', num)
print('num = ', num*2)

num1 = int(input("숫자입력 : "))
print('num1 = ', num1*2)


num2 = float(input("숫자입력 : "))
result = num1 + num2
print('result = ', result)


help(print)


print("value =", 10+20+30+40+50)

print("010", "1234", "5678", sep="-")
print("value=", 10, end=", ")
print("value=", 20)


print("원주율 = ", format(3.14159, "5.8f"))
print("원주율 = ", format(3.14159, "8.3f"))
print("금액 = ", format(1000, "5d"))
print("금액 = ", format(12500000, "3,d"))
print("금액 = ", "125,000")


name = "홍길동"
age = 35
price = 125.456
print("이름 : %s, 나이 : %d, data = %.2f" %(age, age, age))


#print("{}" .format(값)) -> 중괄호 안에 값은 index값임 ㅇㅋ? ㅇㅋ
print("이름 : {}, 나이 : {}, data={}".format(name, age, price))
print("이름 : {0}, 나이 : {1}, data={2}".format(name, age, price))



