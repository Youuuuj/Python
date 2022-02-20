# 유준

# 1

lst = [90, 25, 67, 45, 80]

for i in lst :
    if i >= 60:
        print("합격입니다.")
    else:
        print('불합격')


# 2
def Quest(x,y,z):
    if x < 40 or y < 40 or z < 40 :
        print('불합격')
    elif x + y + z < 180 :
        print('불합격')
    else:
        print('합격')


kor = int(input('국어 점수 : '))
eng = int(input('영어 점수 : '))
mat = int(input('수학 점수 : '))


Quest(kor, eng, mat)


# 3
def Food(x):
    ticket = x

    def getx() :
        return ticket

    def In(money) :
        nonlocal ticket
        print('돈을 넣어주세요. : ', money)

        while True:
            if money > 5000 :
                result = money - 5000
                print("거스름돈 : ", result)
                ticket -= 1

            elif money == 5000 :
                ticket -= 1

            else :
                print("돈이 적습니다")

            if ticket <= 0 :
                print("판매중지")
            break

    return getx, In


get, In = Food(1)

print("티켓 수 : ",get())
In(8000)
print("티켓 수 : ",get())
In(4000)



