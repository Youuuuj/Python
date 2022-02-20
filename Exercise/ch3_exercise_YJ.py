#1 - A

kg = int(input("짐의 무게는 얼마입니까? ", ))

if kg >= 10 :
    print("수수료는 ", format(10000, "3,d"), "원 입니다.")
else kg < 10 :
    print("수수료는 없습니다.")

#1 - B

kg2 = int(input("짐의 무게는 얼마입니까? ", ))

if kg2 < 10 :
    print("수수료는 없습니다.")

else kg2 >= 10 :
    수수료 = int(kg2 / 10)
    print("수수료는 ", format(수수료*10000, "3,d"), "원 입니다.")



#2

import random

print('>>숫자 맞추기 게임<< ')
com = random.randint(1, 10)

while True :
    my = int(input('예상 숫자를 입력하시오 : '))

    if my == com :
        print('~~ 성공 ~~')
        break

    elif my > com :
        print('더 큰 수 입력')

    else my < com :
        print('더 작은 수 입력')




#3
#num = range(3,100,6)

#for n in num :
    #print(n, end = " ")


#print("누적합 = ", sum(num))


cnt = tot = 0

print("수열 =", end = ' ')
for n in range(1, 101):
    cnt += 1
    if cnt % 3 == 0 and cnt % 2 != 0 :
        print (cnt, end=' ')
        tot += cnt

print("누적합= %d" %tot)


#4
multiline = """안녕하세요. 파이썬 세계로 오신걸
환영합니다.
파이썬은 비단뱀 처럼 매력적인 언어입니다."""


doc = []
word = []

for line in multiline.split("\n"):
    doc.append(line)

print(doc)
    for w in line.split(" "):
        word.append(w)
        print(w)

print(word)
print("단어 개수 : ", len(word))