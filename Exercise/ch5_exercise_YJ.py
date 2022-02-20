#유준

#1
def StarCount(height) :
    if height == 1:
        print('*')
        return 1
    else:
        result = height + StarCount(height-1)
        print('*' * height, end='\n')
        return result


height = int(input("height : "))
print('stat 개수 : %d'%StarCount(height))





#2
def bank_account(bal):
    balance = bal  # 잔액초기화(1000)

    def getBalnce() :  # 잔액확인(getter)
        return balance

    def deposit(money) :  # 입금하기(setter)
        nonlocal balance
        balance += money
        print('입급액을 입력하세요 : ', money)
        print(money,"원 입금 후 잔액은",balance,"원 입니다.")

    def withdraw(money) :  # 출금하기(setter)
        nonlocal  balance
        print("출금액을 입력하세요 : ", money)
        if balance - money >= 0 :
            balance -= money
            print(money,"원 출금 후 잔액은",balance,"원 입니다.")
        else :
            print("잔액이 부족합니다")

    return getBalnce, deposit, withdraw

bal = int(input("최초 계좌의 잔액을 입력하세요 : "))

ce, it, aw = bank_account(bal)

print("현재 계좌 잔액은", ce(),"원 입니다.")
it(15000)
ce()
aw(3000)
ce()
aw(14000)


# 3
def Factorial (n) :
    if n == 1:
        return 1
    else :
        result = n * Factorial(n-1)
        return result

result_fact = Factorial(5)
print('팩토리얼 결과:', result_fact)