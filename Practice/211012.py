# 단어 출현 빈도수 구하기

# (1) 단어 데이터 셋
charset = ['abc', 'code', 'band', 'band', 'abc']  # 리스트 형식
wc = {}  # 셋, 딕트 둘중 하나 비순서 자료구조 사용하겠다.

# (2) get()함수를 이용 : key이용 value 가져오기
for key in charset:
    wc[key] = wc.get(key, 0) + 1  # get 이용
print(wc)




# 자료 구조 복제

# (1) 얕은 복사 : 주소 복사(내용, 주소 동일)
name = ['홍길동', '이순신', '강감찬']
print('name address =', id(name))

name2 = name
print('name2 address =', id(name2))

print(name)
print(name2)

# 원본수정 - 같은 주소값을 갖고 있기때문에 사본을 변경해도 원본도 같이 변경되어짐.
name2[0] = '김길동'
print(name)
print(name2)

# (2) 깊은 복사 : 내용 복사(내용 동일, 주소 다름)
import copy
name3 = copy.deepcopy(name)
print(name)
print(name3)

print('name address =', id(name))
print('name3 address =', id(name3))

# 원본수정 - 깊은 복사는 원본과 사본의 주소가 다르기 때문에 원본을 수정해도 사본은 변경되지 않음.
name[1] = '이순신 장군'
print(name)
print(name3)



# 알고리즘(Algorithm)
# 어떤 문제를 해결하기 위한 단계적 절차.
# 조건 : 1.입력, 2.출력, 3.명백성, 4.유한성, 5.유효성

# 최댓값/최솟값(max/min)
# (1) 입력 자료 생성
import random
dataset = []
for i in range(10):
    r = random.randint(1, 100)
    dataset.append(r)

print(dataset)

# (2) 변수 초기화
vmax = vmin = dataset[0]

# (3) 최대값/최솟값 구하기
for i in dataset:
    if vmax < i:
        vmax = i
    if vmin > i:
        vmin = i

# (4) 결과 출력
print('max = ', vmax, 'min = ', vmin)



# 정렬(sort) : 전체자료의 원소를 일정한 순서로 나열하는 알고리
# 작은수에서 큰수로 -> 오름차순
# 큰수에서 작은수로 -> 내림차순


# 선택 정렬 알고리즘 예 : 각 회전이 종료 될 때마다 전체 원소 중 가장작은 값이 왼쪽의 첫번째로 온다.

# (1) 오름차순 정렬
dataset = [3, 5, 1, 2, 4]
n = len(dataset)

for i in range(0, n-1):  # 1 ~ n-1
    for j in range(i+1, n):  # i+1 ~ n
        if dataset[i] > dataset[j]:
            tmp = dataset[i]
            dataset[i] = dataset[j]
            dataset[j] = tmp
    print(dataset)

print(dataset)

# (2) 내림차순 정렬
dataset = [3, 5, 1, 2, 4]
n = len(dataset)

for i in range(0, n-1):  # 1 ~ n-1
    for j in range(i+1, n):  # i+1 ~ n
        if dataset[i] < dataset[j]:
            tmp = dataset[i]
            dataset[i] = dataset[j]
            dataset[j] = tmp
    print(dataset)

print(dataset)


# 검색(search) : 순차검색과 이진검색 있음. 이진검색 알고리즘이 더 빠름
# 이진검색은 자료가 정렬되어 있어야 한다는 전제조건 있음
# 정렬에 필요한 시간 복잡도 까지 고려해서 알고리즘 선택해야함.

# 이진 검색 알고리즘 수행 과정
# mid 위치를 기준으로 오른쪽 절반은 검색대상에서 제외 시키기 위해서는 end = mid - 1
# mid 위치를 기준으로 왼쪽 절반은 검색대상에서 제외 시키기 위해서는 start = mid + 1
# 1 3 5 7 9
# start = 1, end = 5  # 여기서 1과 5는 인덱스가 아니라 원소의 위치
# mid = (start + end)/2 -> 1+5/2 -> mid = 3
# end = mid-1 -> 3-1 = 2
# mid = (start + end)/2 ->  1+2 / 2 -> 1(1.5)
# start = mid+1 -> 1+1 = 2
# mid = (start + end)/2 -> 2+2 / 2 -> 2


# 이진 검색 알고리즘 예
dataset = [5, 10, 18, 22, 35, 55, 75, 103]
value = int(input("검색할 값 입력 : "))

low = 0  # start 위치
high = len(dataset) - 1  # end위치
loc = 0
state = False  # 상태변수

while (low <= high):
    mid = (low + high) // 2

    if dataset[mid] > value:  # 중앙값이 큰 경우
        high = mid -1
    elif dataset[mid] < value:  # 중앙값이 작은 경우
        low = mid + 1
    else:  # 찾은경우
        loc = mid
        state = True
        break  # 반복 exit

if state:
    print('찾은 위치 : %d' %(loc+1))
else:
    print('찾는 값이 없습니다.')


