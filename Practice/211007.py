oneline = "this is one line string"
print(oneline)


multiline = '''this is
multi line
string'''
print(multiline)


multiline2 = "this is \nmulti line \nstring"
print(multiline2)

a = "PYTHON"

print(a[-7])


print(oneline)

print("문자열 길이 : ", len(oneline))

print(oneline[0:8])
print(oneline[:4])
print(oneline[:])
print(oneline[::2])
print(oneline[0:-1:2])
print(oneline[-6:-1])

print(oneline[5:0])

subString = oneline[-11:]
print(subString)

oneLine = "this is one line string"

print('t 글자수 : ', oneLine.count('t'))
print(oneLine)

print(oneLine.startswith('this'))
print(oneLine.startswith('that'))

print(oneLine.replace('this','that'))

multiLine = """this is 
multi line
string"""

sent = multiLine.split('\n')
print('문장 : ', sent)

print(oneLine)


words = oneLine.split(' ')
print('단어 : ', words)


sent2 = ','.join(words)
print(sent2)


oneLine = "this is 'one' \"line\" 'string'"
print(oneLine)

print('escape 문자차단')
print('\n출력 이스케이프 문자')

print('\\n출력 이스케이프 기능 차단 1')
print(r'\n출력 이스케이프 기능 차단 1')

print('path =', 'C:\Python\test')
print('path =', 'C:\Python\\test')
print('path =', r'C:\Python\test')


var = 10
if var >= 5 :
    print('var = ', var)
    print('var는 5보다 크다.')
    print('조건이 참인 경우 실행')

print('항상 실행')

score = int(input('점수 입력(0~100) : '))
if score >= 85 and score <=100 :
    print('우수')
else :
    if score >= 70 :
        print('보통')
    else :
        print('저조')

score = int(input("점수입력 : "))
grade = ""

if score >= 85 and score <= 100 :
    grade = "우수"
elif score >= 70 :
    grade = "보통"
else :
    grade = "저조"

print("당신의 점수는 %d이고, 등급은 %s"%(score, grade))


num = 9
result = 0

if num >= 5 :
    result = num * 2
else :
    result = num + 2
print('result = ', result)


result2 = num * 2 if num >= 5 else num + 2
print('result2 = ', result2)



# (1) 카운터와 누적변수
cnt = tot = 0
while cnt < 5 :
    cnt += 1
    tot += cnt
    print (cnt, tot)


cnt = tot = 0
dataset = []

while cnt < 100 :
    cnt += 1
    if cnt % 3 == 0:
        tot += cnt
        dataset.append(cnt)

print('1 ~ 100 사이 3의 배수 합 = %d' % tot)
print('dataset =', dataset)


numData = []

while True:
    num = int(input("숫자 입력 : "))

    if num % 10 == 0 :
        print("프로그램 종료")
        break
    else :
        print(num)
        numData.append(num)

print(numData)

import random
help(random)

help(random.random)

r = random.random()
print ("r= ", r)

cnt = 0
while True:
    r = random.random()
    print(random.random())
    if r < 0.01:
        break
    else:
        cnt += 1

print('난수개수 = ', cnt)

help (random. randint)


names = ["홍길동", "이순신", "유관순"]
print(names)
print(names[2])

if '유관순' in names :
    print ("유관순 있음")
else:
    print("유관순 없음")

idx = random.randint(0, 2)
print (names[idx])

i = 0
while i < 10:
    i += 1
    if i == 3:
        continue
    if i == 6:
        break
    print(i, end=" ")

string = "홍길동"
print(len(string))
for s in string :
    print(s)

lstset = [1,2,3,4,5]

for e in lstset :
    print('원소 : ', e)

num1 = range(10)
print('num1 : ', num1)
list(range(1,10))


num2 = range(1, 10)
print('num2 : ', num2)

num3 = range(1, 10, 3)
print('num3 : ', num3)


for n in num1 :
    print(n, end = ' ')

for n in num2 :
    print(n, end = ' ')

for n in num3:
    print(n, end=' ')

lst = []
for i in range(10) :
    r = random.randint(1,10)
    lst.append(r)

print('lst= ', lst)

for i in range(10) :
    print(lst[i] * 0.25)

