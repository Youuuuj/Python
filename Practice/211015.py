# 클래스와 객체

# 클래스는 속성과 행위로 구성된다.
# 객체는 클래스에 의해서 만들어지는 결과물. 자료 + 메소드


# 함수와 클래스의 예
# (1) 함수정의
def calc_func(a, b):
    # 변수 선언
    x = a  # 10
    y = b  # 20

    def plus():
        p = x+y
        return p

    def minus():
        m = x-y
        return m

    return plus, minus

# (2) 함수호출
p, m = calc_func(10, 20)
print('plus = ', p())
print('minus = ', m())


# (3) 클래스정의
class calc_class :
    # 클래스 변수 : 자료저장
    x = y = 0

    # 생성자 : 객체 생성 + 멤버변수 초기화
    def __init__(self,a,b):
        self.x = a  # 10
        self.y = b  # 20

    # 클래스 함수
    def plus(self):
        p = self.x + self.y  # 변수 연산
        return p

    # 클래스 함수
    def minus(self):
        m = self.x - self.y  # 변수 연산
        return m

# (4) 객체 생성
obj = calc_class(10, 20)

# (5) 멤버 호출
print('plus = ', obj.plus())
print('minus = ', obj.minus())




# 클래스 구성요소 예
class Car :
    # (1) 멤버변수
    cc = 0  # 엔진 cc
    door = 0  # 문짝 개수
    carType = None  # null

    # (2) 생성자
    def __init__(self,cc,door,carType):
        # 멤버 변수 초기화
        self.car = cc
        self.door = door
        self.carType = carType

    # (3) 메소드
    def disply(self):
        print("자동차는 %d cc이고, 문짝은 %d개, 타입은 %s"%(self.car, self.door, self.carType))

# (4) 객체생성
car1 = Car(2000, 4, "승용차")  # 객체생성 + 초기화
car2 = Car(3000, 5, "SUV")

# (5) 멤버 호출 : object.member
car1.disply()
car2.disply()




# 생성자 : 생성자의 역할은 객체 생성 시 멤버 변수에 값을 초기화 하는 역할.
# 만약 초기화 작업이 필요없으면 클래스 설계 시 생략 가능.  ->  파이썬 자체에서 기본 생성자 제공해줌.

# 생성자 예
# (1) 생성자 이용 멤버 변수 초기화
class multiply :
    #멤버변수
    x = y = 0

    # 생성자 : 초기화
    def __init__(self, x, y):  # 객체만 생성
        self.x = x
        self.y = y

    # 메소드
    def mul(self):
        mu = self.x * self.y
        return mu

obj = multiply(10 ,20)  # 생성자
print("곱셈 = ", obj.mul())


# (2) 메소드 이용 멤버변수 초기화
class multiply2 :
    # 생성자 없음 : 기본 생성자 제공. 지워도 실행가능
    def __init__(self):
        pass
    # 메소드 : 멤버변수 초기화
    def data(self, x, y):
        self.x = x
        self.y = y
    # 메소드 : 곱셈
    def mul(self):
        mu = self.x * self.y
        return mu


obj = multiply2()  # 기본 생성자
obj.data(10, 20)  # 동적 멤버변수 생성
print("곱셈 = ", obj.mul())




# self : 클래스를 구성하는 멤버들 즉 멤버변수와 메소드를 호출하는 역할을 한다.
class multiply3 :
    # 멤버변수 없음
    # 생성자 없음

    # 동적 멤버변수 생성/초기화
    def data(self, x, y):
        self.x = x
        self.y = y

    # 곱셈 연산
    def mul(self):
        result = self.x * self.y
        self.display(result)  # 메서드 호출

    def display(self, result):
        print("곱셈 : %d"%(result))


obj = multiply3()
obj.data(10,20)
obj.mul()



# 클래스 멤버 : 클래스 이름으로 호출 할 수 있는 클래스 변수와 클래스 메소드를 말한다.
# 클래스 이름으로 호출할 수 있기 때문에 클래스 멤버를 호출하기 위해서 객체를 생성할 필요는 없다.
# 클래스 메소드는 cls라는 기본인수 사용하고, @classmethod라는 함수 장식자를 이용하여 선언.


class DatePro:
    # (1) 멤버 변수
    content = "날짜 처리 클래스"

    # (2) 생성자
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    # (3) 객체 메소드(instance method)
    def display(self):
        print("%d-%d-%d"%(self.year,self.month,self.day))

    # (4) 클래스 메소드(class method)
    @classmethod  # 함수장식자
    def date_string(cls, dateStr):  # '19951025'
        year = dateStr[:4]
        month = dateStr[4:6]
        day = dateStr[6:]

        print(f"{year}년 {month}월 {day}일")


# (5) 객체 멤버
date = DatePro(1995, 10, 25)
print(date.content)
print(date.year)
date.display()

# (6) 클래스 멤버
print(DatePro.content)
print(DatePro.year)
DatePro.date_string('19951025')




# 캡슐화 : 자료와 알고리즘이 구현된 함수를 하나로 묶고 공용 인터페이스만으로 접근을 제한하여 객체의 세부내용을 외부로부터 감추는 기법.
class Account :
    # (1) 은닉 멤버변수
    __balance = 0
    __accName = None
    __accNo = None

    # (2) 생성자 : 멤버변수 초기화
    def __init__(self, bal, name, no):
        self.__balance = bal
        self.__accName = name
        self.__accNo = no

    # (3) 계좌정보 확인 : Getter
    def getBalance(self):
        return self.__balance, self.__accName, self.__accNo

    # (4) 입금하기 : Setter
    def deposit(self, money):
        if money < 0:
            print('금액 확인')
            return  # 종료(Exit)
        self.__balance += money

    # (5) 출금하기 : Setter
    def withdraw(self, money):
        if self.__balance < money :
            print('금액 부족')
            return  # 종료(Exit)
        self.__balance -= money


# (6) object 생성
acc = Account(1000, '홍길동', '125-152-4125-41')  # 생성자

# (7) Getter 호출
acc.__balance  # 오류
bal = acc.getBalance()
print('계좌정보 : ', bal)

# (8) Setter 호출
acc.deposit(10000)  # 10000원 입금
acc.withdraw(30000)  # 30000원 출금
bal = acc.getBalance()
print('계좌정보 : ', bal)




# 상속 : 자식클래스는 부모클래스의 멤버(멤버변수, 메서드)를 상속받음. 그러나 생성자는 상속의 대상이 아님.

# (1) 부모 클래스
class Super:
    # 생성자 : 동적 멤버 생성
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # 메서드
    def display(self):
        print("name : %s, age: %d"%(self.name, self.age))

sup = Super('부모', 55)
sup.display()  # 부모 멤버 호출

# (2) 자식 클래스
class Sub(Super):
    gender = None  # 자식 멤버

    # (3)-1 생성자
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender

    # (3)-2  생성자
    def __init__(self, name, age, gender):
        Super.__init__(self, name, age)
        # super().__init__(self, name, age)
        #self.name = name
        #self.age = age
        self.gender = gender

    # (4) 메서드 확장
    def display(self):
        print("name : %s, age : %d, gender : %s"%(self.name, self.age, self.gender))

sub = Sub('자식', 25, '여자')
sub.display()  # 자식 멤버 호출



# super 클래스 : 자식 클래스에서 부모 클래스의 생성자가 필요한 경우 사용.

# (1) 부모 클래스
class Parent :

    # 생성자 : 객체 + 초기화
    def __init__(self, name, job):
        self.name = name
        self.job = job

    # 멤버함수(method)
    def display(self):
        print('name : {}, job : {}'.format(self.name, self.job))

# 부모 클래스 객체 생성
P = Parent('홍길동','회사원')
P.display()


# (2)-1 자식 클래스
class Children(Parent):
    gender = None

    # (3) 자식 클래스 생성자
    def __init__(self, name, job, gender):
        # 부모 클래스 생성자 호출
        super().__init__(name, job)
        self.gender = gender

    # 멤버함수
    def display(self):
        print('name : {}, job : {}, gender : {}'.format(self.name,self.job,self.gender))

chil = Children("이순신", "해군 장군", "남자")
chil.display()


# (2)-2 자식 클래스
class Children(Parent):
    gender = None

    # (3)-2 자식 클래스 생성자
    def __init__(self, name, job, gender):
        # 부모 클래스 생성자 호출
        Parent.__init__(self, name, job)
        self.gender = gender

    # 멤버함수
    def display(self):
        print('name : {}, job : {}, gender : {}'.format(self.name, self.job, self.gender))


chi = Children("이순신", "해군 장군", "남자")
chi.display()

