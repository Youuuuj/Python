
#유준


#1

lst = [10,1,5,2]  # 리스트 생성

#1-1
result1 = lst * 2
print("단계1 : ", result1)

#1-2
lst2 = lst[0] * 2
result1.append(lst2)
print("단계2 : ", result1)

#1-3
result2 = result1[1:9:2]
print("단계3 : ", result2)


#2-A
size = int(input('vector 수 : '))

lst3 = []
for i in range(size):
    lst3.append(int(input()))

print("vector의 크기 : ",len(lst3))

#2-B
size2 = int(input('vector 수 : '))

lst4 = []
for i in range(size2):
    lst4.append(int(input()))



#3
message = ['spam', 'ham', 'spam', 'ham', 'spam']

#3-A
dummy = [1 if i=='spam' else 0 for i in message]
print(dummy)

#3-B
spam_list = [i for i in message if i == 'spam']
print(spam_list)



#4
position = ['과장', '부장', '대리', '사장', '대리', '과장']

#4-1
sp = set(position)
lp = list(sp)
print(lp)

#4-2
wc = {}
for key in position:
    wc[key] = wc.get(key, 0) + 1

print(wc)