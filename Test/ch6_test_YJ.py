# 유준

# 11

f = open("foo.txt", mode='w', encoding='utf-8')
f.write('Life is too short, you need python')
f.close()

# f.close 없이 파일 오픈
try:
    with open('foo.txt', mode='r', encoding='utf-8') as f:
        line = f.readline()
        print(line)

except Exception as e:
    print('Error : ', e)

finally:
    pass




# 12
class bank_account :
    # (1) 멤버변수
    balance = 0

    # (2) 생성자
    def __init__(self,balance):
        # 멤버 변수 초기화
        self.balance = balance
        print('현재 잔액 : {}원'.format(self.balance))

    # (3) 메소드 - 입금
    def deposit(self, money):
        print("얼마를 입금하시겠습니까? : {}원".format(money))
        if money > 0:
            self.balance += money
            print('현재 잔액 : {}원'.format(self.balance))

        else:
            print('금액 재입력')

    # (4) 메소드 - 출금
    def withdraw(self, money):
        print("얼마를 출금하시겠습니까? : {}원".format(money))
        if self.balance >= money :
            self.balance -= money
            print('현재 잔액 : {}원'.format(self.balance))

        elif self.balance < money:
            print('잔액이 부족합니다. 재입력 하십시오')

        else:
            print('금액 재입력')

    # (5) 메소드 - 이자 확인
    def interest(self, interest):
        inte = int(self.balance * interest)
        print('한달의 이자액 : {}원'.format(inte))



B = bank_account(10000)
B.deposit(5000)
B.withdraw(3000)
B.interest(0.02)




