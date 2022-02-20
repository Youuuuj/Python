# 유준

# 1
email = '''hong@12.com
you2@naver.com
12kang@hanmail.com
kimjs@gmail.com'''

from re import findall, match, split

for e in email.split(sep='\n') :
    mat = match('^[a-z]\\w{3,}@[a-z]\\w{2,}.[a-z]\\w{,2}', e)

    if mat :
        print(e)


#    x1 = findall('^[a-z]\\w{3,}', e)
#    x2 = findall('@[a-z]\\w{2,}', e)
#    x3 = findall('[.][a-z]\\w{,2}', e)


# 2
from re import findall

emp = ['2014홍길동220', '2002이순신300', '2010유관순260']

# 함수정의
def name_pro(emp):
    names = []
    for i in emp :
        name = findall('[가-힣]{3}', i)
        names += name
    return names


names = name_pro(emp)
print('names = ', names)


# 2-1
def name_pro(emp):
    names = []

    for i in range(len(emp)):
        name = findall('[가-힣]{3}', emp[i])
        names += name
    return names

names = name_pro(emp)
print('names = ',names)


# 3
from re import findall
from statistics import mean

emp = ['2014홍길동220', '2002이순신300', '2010유관순260']


# 함수정의
def pay_pro(emp):
    pay2 = []

    for i in range(len(emp)):
        pay = findall('[\d]{3}$', emp[i])
        pay2 = list(map(int, pay))
    return mean(pay2)


pays_mean = pay_pro(emp)
print('전체 사원의 급여 평균 : ', pays_mean)



# 4
from re import  findall
from  statistics import mean

emp = ['2014홍길동220', '2002이순신300', '2010유관순260']

# 함수 정의
def pay_pro(x):
    from statistics import mean
    import re

    names = []
    mpay = []
    tpay = []

    for i in range(len(emp)):
        pay = findall('[\d]{3}$', emp[i])
        mpay = list(map(int, pay))
        tpay += mpay
        name = findall('[가-힣]{3}', emp[i])
        names += name

    me = mean(tpay)
    print("전체 급여 평균 : %d" % me)
    print("평균 이상 급여 수령자", end="\n")

    for e in range(len(tpay)):
        if tpay[e] >= 260:
            print("%s ==> %d" % (names[e], tpay[e]))


pay_pro(emp)


# 5
from re import findall, sub

texts = ['AFAB43747,asabag?', 'abTTa $$;a12:2424.','uysfsfA,A1234&***$?']

texts_re1 = [t.lower() for t in texts]
print(texts_re1)

texts_re2 = [sub('[0-9]', '', text) for text in texts_re1]
print(texts_re2)

texts_re3 = [sub('[,.?!:;]', '', text) for text in texts_re2]
print(texts_re3)

spec_str = '[@#$%^&*()]'
texts_re4 = [sub(spec_str, '', text) for text in texts_re3]
print(texts_re4)

texts_re5 = [''.join(text.split()) for text in texts_re4]
print(texts_re5)