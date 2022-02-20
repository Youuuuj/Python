# 유준

#1

num = 100

if num > 100:
    print("100보다 큽니다.")
elif num > 50:
    print("50보다 큽니다.")
else:
    print("50이하입니다.")



#2

for i in range(2,10):
    print("{}단".format(i))

    for j in range(1,10):
    print('%d * %d = %d' %(i, j, i*j))


#3
line = """It ain't over
til it's over
by Yogi Berra"""


#3-1
sents = []
words = []
words2 = []

for sen in line.split():
    sents.append(sen)
    for word in sen.split():
        words.append(word)

print(words)

words2 = ' '.join(words)
print(words2)

#3-2
len(line)
L = (line[16].upper() + line[9:12] + line[16] + line[-10].lower())
print(L)