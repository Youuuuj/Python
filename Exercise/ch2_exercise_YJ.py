#1


su = 5
dan = 800
print("su 주소 :", id(su))
print("dan 주소 :", id(dan))
result = su * dan
print("금액=", result)


#2
x = 2
y = 2.5 * x**2 + 3.3 * x + 6
print("2차 방정식 결과 = ", y)


#3

지방 = input("지방의 그램을 입력하세요 : ")
탄수화물 = input("탄수화물의 그램을 입력하세요 : ")
단백질 = input("단백질의 그램을 입력하세요 : ")
총칼로리 =  int(지방)*9 + int(단백질)*4 + int(탄수화물)*4
print("총칼로리 : ", format(총칼로리, "3,d"), "cal")


#4

word1 = input("첫번째 단어 : ")
word2 = input("두번째 단어 : ")
word3 = input("세번째 단어 : ")
word4 = input("네번재 단어 : ")
#print("==============")

abbr = word1[0]+word2[0]+word3[0]
print("="*15)
print("약자 : " + abbr)

#4-1

abbr2 = word2[-1].upper()+word3[0:1].lower()+word4[0:1].lower()+word1[-2]
print(abbr2)