# Numpy 제공:
# 효율적인 다차원 배열인 ndarray 는 빠른 배열 계산과 유연한 브로드캐스팅 기능 제공
# 반복문을 작성할 필요 없이 전체 데이터 배열을 빠르게 계산할 수 있는 표준 수학함수
# 배열 데이터를 디스크에 쓰거나 읽을 수 있는 도구와 메모리에 적재된 파일을 다루는 도구
# 선형대수, 난수 생성기, 푸리에 변환 기능
# C, C++, Fortran 으로 작성한 코드를 연결할 수 있는 C API

# Numpy 자체는 모델링이나 과학 계산을 위한 기능을 제공하지 않음.
# Numpy 배열과 배열 기반 연산에 대한 이해 후 pandas 같은 배열 기반 도구 사용 시 더 효율적.
# pandas 는 Numpy 에는 없는 시계열 처리 같은 다양한 도메인 특화기능 제공

# Numpy 는 내부적으로 데이터를 다른 내장 파이썬 객체와 구분된 연속된 메모리 블록에 저장.
# 각종 알고리즘은 모두 C 로 작성되어 메모리를 직접 조작할 수 있고 훨씬 더 적은 메모리를 사용.
# 전체 배열에 대한 복잡한 계산을 수행 가능
# --> 대용량 데이터 배열을 효율적으로 다룰 수 있도록 설계

import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision = 4, suppress = True)

import numpy as np
my_arr = np.arange(1000000)
my_list = list(range(1000000))

from datetime import datetime
start1 = datetime.now()
for _ in range(10): my_arr2 = my_arr * 2
print(datetime.now() - start1)

from datetime import datetime
start2 = datetime.now()
for _ in range(10): my_list2 = [x * 2 for x in my_list]
print(datetime.now() - start2)


# 4.1 The NumPy ndarray: A Multidimensional Array Object
# ndarray : N 차원의 배열 객체 대규모 데이터 집합을 담을 수 있는 빠르고 유연한 자료구조
# 배열은 스칼라 원소간의 연산에  사용하는 비슷한 방식 사용

# 배치 계산 처리 방법
import numpy as np
# random data 만들기
data = np.random.randn(4,4)
data

data * 10
data + data

# from numpy import * 를 사용하면 np 입력 불필요.
# but import numpy as np convention 사용

# ndarray : 같은 종류이 데이터를 담을 수 있는 다차원 배열.
# ndarray의 모든 원소는 같은 자료형이어야 한다.
# 모든 배열은 각 차원이 크기를 알려주는 shape이라는 튜플과
# 배열에 저장된 자료형을 알려주는 dtype이라는 객체를 가지고 있다.
data.shape
data.dtype


# 4.1.1 Creating ndarrays
# 배열을 생성하는 가장 쉬운 방법은 array()함수를 이용하는 것
# 순차적인 객체(다른 배열도 포함하여)를 넘겨받고, 넘겨받은 데이터가 들어 있는 새로운 NumPy 배열을 생성

# 리스트 변환 예제
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1

# 같은 길이를 가지는 리스트를 내포하고 있는 순차 데이터는 다차원 배열로 변환 가능
data2 = [[1,2,3,4], [5,6,7,8]]
arr2 = np.array(data2)
arr2

import numpy as np
data = [[1,2,3],[2,3,4]]
array2 = np.array(data)
print('array 2의 타입은:', type(array2))
array3 = np.array([[1,2,3],[2,3,4]])
array3

data3 = [[1,2,3,4,5,6], [7,8,9,10,11,12]]
data4 = [[[1,2,3,4],[4,5,6,4]],[[7,8,9,4],[10,4,11,12]]]
arr3 = np.array(data4)
arr3 = np.array(data3).reshape(4,3)
arr3.ndim  # 차원
arr3.shape  # 행과 열
arr3.size


# data2는 리스트를 담고 있는 리스트
# Numpy배열인 arr2는 해당 데이터로부터 형태를 추론하여 2 차원 형태로 생성된다.
# ndim과 shape속성을 검사하여 확인 가능
arr2.ndim  # 차원
arr2.shape  # 행과 열
arr2.size  # 전체 원소 수
len(arr2)  # 첫번째 차원의 갯수   -  이게 뭐소리야?

# np.array가 생성될 때 자료형은 dtype객체에 저장된다.
arr1.dtype
arr2.dtype

# zeros와 ones는 주어진 길이나 모양에 각각 0과 1이 들어있는 배열 생성
np.zeros(10)
np.zeros((3,6))
np.zeros((2,2,3))  # 차원, 행, 열 순

np.ones(10)
np.ones((2,7))
np.ones((4,1,2))

# empty함수는 초기화되지 않은 배열 생성
np.empty((2,3,2))

# arange는 python의 range 함수의 배열 버전
np.arange(15)

# 표 4-1 배열 생성 함수
# array: 입력 데이터(리스트, 튜플, 배열 또는 다른 순차형 데이터)를 ndarray 로 변환하며 dtype을 명시하지 않은 경우 자료형을 추론하여 저장
# asarray: 입력 데이터를 ndarray 로 변환하지만 입력 데이터가 이미 ndarray 일 경우 복사가 일어나지 않는다.
# arange: 내장 range 함수와 유사. 리스트 대신 ndarray 반환
# ones, ones_like: 주어진 dtype 과 모양을 가지는 배열을 생성하고 내용을 모두 1 로 초기화.
# ones_like 는 주어진 배열과 동일한 모양과 dtype을 가지는 배열을 새로 생성하여 내용을 모두 1 로 초기화
# zeros, zeros_like: 내용을 모두 0 으로 배열
# empty, empty_like: 메모리를 할당하여 새로운 배열을 생성하지만 ones 나 zeros 처럼 값을 초기화하지 않는다.
# full, full_like: 인자로 받은 dtype 과 배열의 모양을 가지는 배열을 생성하고 인자로 받은 값으로 배열을 채운다.
# eye, identity: NxN 크기의 단위행렬 생성


# 4.1.2 Data Types for ndarrays
# dtype 은 ndarray 가 메모리에 있는 특정 데이터를 해석하기 위해 필요한 정보(또는 메타데이터)를 담고 있는 특수한 객체
arr1 = np.array([1, 2, 3], dtype = np.float64)
arr2 = np.array([1, 2, 3], dtype = np.int32)
arr1.dtype
arr2.dtype
arr1
arr2

# 산술테이터의 dtype은 float나 int 같은 자료형의 일므과 하나아ㅢ 원소가 차지하는 비트 수로 이루어진다.
# 파이썬의 float객체에서 사용되는 표준 배정밀도 부동소수점같은 8바이트 또는 64비트로 이루어지는데 NumPy도 float64로 표현된다.

## Numpy 자료형
# 자료형 자료형 코드 설명
# -------------------------------------------------------------------------------
# int8, uint8 i1, u1 부호가 있는 8 비트(1 바이트) 정수형과 부호가 없는 8 비트 정수형
# int16, uint16 i2, u2 부호가 있는 16 비트 정수형과 부호가 없는 16 비트 정수형
# int32, uint32 i4, u4 부호가 있는 32 비트 정수형과 부호가 없는 32 비트 정수형
# int64, unit64 i8, u8 부호가 있는 64 비트 정수형과 부호가 없는 64 비트 정수형
# float16 f2 반정밀도 부동소수점
# float32 f4 or f 단정밀도 부동소수점. C 언어의 float 형과 호환
# float64 f8 or d 배정밀도 부동소수점. C 언어의 double 형과 파이썬의 float 객체와 호환
# float128 f16 or g 확장정밀도 부동소수점
# complex64 c8 각각 2 개의 32, 54, 128 비트 부동소수점형을 가지는 복소수
# complex128 c16
# complex256 c32
# bool ? True 와 False 값을 가지는 불리언형
# object 0 파이썬 객체형
# string_ S 고정길이 아스키 문자열형(각 문자는 1 바이트, 길이가 10 인 문자열 dtype 은 S10)
# unicode_ U 고정길이 유니코드형 (예, U10)
# -------------------------------------------------------------------------------


# astype 메서드 사용으로 배열의 dtype을 다른 형으로 변환(캐스팅)
arr = np.array([1,2,3,4,5])
arr.dtype
arr
float_arr = arr.astype(np.float64)
float_arr.dtype
float_arr

# 부동소수점수를 정수형 dtype로 변환하면 소수점 아래 자리는 버려진다.
arr = arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr
arr.astype(np.int32)

# astype을 사용하여 숫자 형식의 문자열을 담고 있는 배열을 숫자로 변환
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype = np.string_)
numeric_strings
numeric_strings.astype(float)
# 주의! NumPy에서 문자열 데이터는 고정 크기를 가지며 경고 없이 입력을 임의로 잘라낼 수 있으므로
# numpy.string_형을 이용할 떄는 주의해야함
# .astype(float)에서 float로 입력해도 NumPy에서 알맞은 dtype로 설정

# 다른 배열의 dtype속성 이용
int_array = np.arange(10)
int_array
calibers = np.array([.22,.270,.357,.380,.44,.50], dtype = np.float64)
calibers.dtype
flo_arr = int_array.astype(calibers.dtype)
flo_arr.dtype

# 축약 코드 사용
empty_uint32 = np.empty(8, dtype = 'u4')  # 부호가 없는 32비트 정수형
empty_uint32
# * astype를 호출하면 새로운 dtype이 이전 dtype과 동일해도 항상 새로운 배열을 생성(데이터를 복사)한다.
empty_uint32.astype('u4')


# 4.1.3 Arithmetic with NumPy Arrays
# 배열의 중요한 특징은 for 문을 작성하지 않고 데이터를 일괄 처리 가능.
# --> '벡터화' : 같은 크기의 배열 간의 산술 연산은 배열의 각 원소 단위로 적용
arr = np.array([[1.,2., 3.],[4.,5.,6.]])
arr
arr * arr
arr - arr

# 스칼라 인자가 포함된 산술 연산의 경우 배열 내의 모든 원소에 스칼라 인자가 적용된다.
1 / arr
arr ** 0.5

# 같은 크기를 가지는 배열 간의 비교 연산은 불리언 배열을 반환
arr2 = np.array([[0., 4., 1.], [7.,2.,12.]])
arr2
arr2 > arr
# 브로드 캐스팅 : 크기가 다른 배열 간의 연산


# 4.1.4 Basic Indexing and Slicing
# 데이터의 부분집합이나 개별 요소를 선택하는 방법
arr = np.arange(10)
arr
arr[5]
arr[5:8]
arr[5:8] = 12
arr

# 스칼라값 12를 대입하면 선택 영역 전체로 전파(브로드캐스트)
# 리스트와 중요한 차이점 : 배열 조각은 원본 배열이 뷰(View)
# 즉 데이터는 복사되지 않고 뷰에 대한 변경은 그대로 원본 배열에 반영
# arr 배열의 슬라이스 생성
arr_slice = arr[5:8]
arr_slice

# arr_slice 값을 변경하면 원래 배연인 arr의 값도 바뀌어 있음을 확인할 수 있다.,
arr_slice[1] = 12345
arr

# [:]로 슬라이스하면 배열의 모든 값을 할당
arr_slice[:] = 620
arr

# * 뷰 대신 ndarray 슬라이스의 복사본을 얻고 싶다면 arr[5:8].copy() 사용
arr_copy = arr[5:8].copy()
arr_copy[1] = 12345
arr_copy
arr

# 다차원 배열 중 2차원 배열에서 각 색인에 해당하는 요소는 1차원 배열
arr2d = np.array([[1,2,3,],[4,5,6,],[7,8,9]])
arr2d[2]

# 개별요소는 재귀적으로 접근해야 한다. 또는 콤마로 구분되 ㄴ색인 리스트를 넘긴다.
arr2d[0][2]
arr2d[0,2]

# 다차원배열에서 마지막 색인을 생략하면 반환되는 객체는 상위 차원의 데이터를 포함하고 있는 한 차원 낮은 ndarray가 된다.
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
arr3d[0]

# arr3d[0]에는 스칼라값과 배열 모두 대입 가능
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d

# arr3d[1,0]은 (1,0)으로 색인되는 1차원 배열과 그 값 반환
arr3d[1,0]

# 이 값은 아래의 결과와 동일
x = arr3d[1]
x
x[0]

arr3d[1][0]

# indexing with slices
# ndarray는 익숙한 문법으로 슬라이싱
arr
arr[1:6]

# arr2d의 경우 슬라이싱 방법이 상이
arr2d
arr2d[:2]

# 슬라이스 축을 따라 선택 영역 내의 요소를 선택한다.
# arr2d[:2]는 'arr2d의 시작부터 두 번째 행까지의 선택'이라고 이해할 것.

# 다차원 슬라이싱
arr2d[:2, 1:]

# 정수 색인과 슬라이스를 함께 사용해서 한 차원 낮은 슬라이스를 얻을 수 있다.
# 두번째 로우에서 처음 두 컬럼만 선택
arr2d[1, :2]
# 처음 두 로우에서 세번쨰 컬럼만 선택
arr2d[:2, 1]
# :만 사용하면 전체 축을 선택한다는 의미이므로 원래 차원의 슬라이스를 얻게 된다.
arr2d[:, :1]
# 슬라이싱 구문에 값을 대입하면 선택 영역 전체의 값이 대입된다.
arr2d[:2, 1:] = 0
arr2d


# 4.1.5 Boolean Indexing
# 중복된 이름을 포함한 배열. randn 함수를 이용하여 표준 정규분포 데이터 생성
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4) # 7x4 array
names
data

# names와 'Bob'문자열 비교하면 불리언 배열 반환
names == 'Bob'
# 이 Boolean 배열을 배열의 색인으로 사용 가능
data[names == 'Bob']
data[[0,3],]
data[[0,3]]
# *불리언배열은 반드시 색인하려는 축의 길이와 동일한 길이를 가져야함.
# 하지만 동일한 길이가 아니더라고 error은 발생하지 않으므로 주의!
# names == 'Bob'인 로우에서 2: 컬럼선택
data[names == 'Bob', 2:]  # 행 인덱스 0,3 에서 열 인덱스 2:
data[names == 'Bob', 3]

data[[0,3], 2:]
data[[0,3], 3]

# ~연산자는 반대조건
names != 'Bob'
data[~(names == 'Bob')]

cond = names == 'Bob'
data[~cond]

# &조건과 |조건
# Boolean배열에서 and와 or사용 불가능 따라서 &와 |를 사용해야 함
mask = (names == 'Bob') | (names == 'Will')
mask
data[mask]
# 배열에 불리언 색인을 이용해서 데이터를 선택하면 반환되는 배열의 내용이 바뀌지 않더라도 항상 데이터 복사가 발생

# data에 저장된 모든 음수를 0으로 대입
data[data < 0] = 0
data

# 1차원 불리언 배열을 사용해서 전체 로우나 컬럼을 선택하는 것은 쉽게 할 수 있다.
data[names == 'Joe'] = 7
data

# * 2차원 데이터에 대한 연산은 pandas를 이용해서 처리하는것이 편리



# 4.1.6 Fancy Indexing
# Fancy Indexing 은 정수 배열을 사용한 색인을 설명하기 위함.
# 8x4 배열
arr = np.empty((8, 4))# 8x4 array
for i in range(8):
 arr[i] = i
arr

# 특정순서로 row 선택시 해당 정수가 담긴 ndarray나 리스트를 넘김
arr[[4,3,0,6]]
# 색인으로 음수를 사용하면 끝에서부터 로우 선택
arr[[-1, -5, -7]]

# 다차원 색인 배열을 넘기는 것은 다르게 동작. 각각의 색인 튜플에 대응하는 1 차원 배열이 선택됨.
arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]] # index [1,0], [5, 3], [7, 1], [2, 2]
# 배열이 몇차원이든 팬시 색인의 결과는 항상 1차원

# 행렬의 행(로우)와 열(컬럼)에 대응하는 사각형 모양의 값이 선택되기 위해서는 아래와 같이 코딩
arr[[1, 5, 7, 2]][:,[0, 3, 1, 2]] # row index 선택, column index 순서로 배열

# 4.1.7 Transposing Arrays and Swapping Axes
# 배열 전치는 데이터를 복사하지 않고 데이터의 모양이 바뀐 뷰를 반환하는 특별한 기능
# ndarray 는 transpose 메서드와 T 라는 이름의 특수한 속성 보유
arr = np.arange(15).reshape((3, 5))
arr
arr.T

# 행렬의 계산은 np.dot 를 이용
arr = np.random.randn(6, 3)
arr
np.dot(arr.T, arr)

# 다차원 배열의 경우 transpose메서드는 튜프를 축 번호를 받아서 치환한다,
arr = np.arange(16).reshape((2,2,4))
arr
arr.transpose((1,0,2))

# 예제
a = np.ones((2,3,4))
a
np.transpose(a, (1,0,2)) # reshape 된 arrary 의 index 배열. (1,0,2) --> (3, 2, 4)
np.transpose(a,(1,0,2)).shape
np.transpose(a,(2,1,0)) # reshape 된 arrary 의 index 배열. (2, 1, 0) --> (4, 3, 2)
np.transpose(a,(2,1,0)).shape

# 두 개의 축 번호를 받아서 배열을 뒤 바꾼다.
arr
arr.swapaxes(1,2)
# swapaxes 도 데이터를 복사하지 않고 원래 데이터에 대한 뷰를 반환


## 4.2 Universal Functions: Fast Element-Wise Array Functions
# 유니버설 함수 ufunc: ndarray 안에 있는 데이터 원소별로 연산을 수행하는 함수
# ufunc 함수는 sqrt 나 exp 같은 간단한 변형을 전체 원소에 적용 가능

arr = np.arange(10)
arr
# 단항 유니버설 함수
np.sqrt(arr)
np.exp(arr)

# 이항 유니버설 함수
x = np.random.randn(8)
y = np.random.randn(8)
x
y
np.maximum(x, y)

# 여러개의 배열을 반환하는 유니버설 함수
# modf 는 파이썬 내장 함수인 divmod 의 벡터화 버전. 분수를 받아서 몫과 나머지를 함께 반환
arr = np.random.randn(7) * 5
arr
remainder, whole_part = np.modf(arr)
remainder
whole_part

# 선택적으로 out인자 사용하여 계산 결과를 별도 저장
arr
np.sqrt(arr)
np.sqrt(arr, arr)
arr

# 단항 유니버설 함수
# 함수 설명
# abs, fabs 각 원소(정수, 부동소수점수, 복소수)의 절대값. 복소수가 아닌경우 빠른 연산을 위해 fabs 사용
# sqrt 각 원소의 제곱근 계산. arr**0.5 와 동일
# square 각 원소의 제곱을 계산. arr**2 와 동일
# exp 각 원소에서 지수 e**x 를 계산
# log, log10, log2, log1p 각각 자연로그, 로그 10, 로그 2, 로그(1+x)
# sign 각 원소의 부호를 계산
# ceil 각 원소의 소수자리 올림
# floor 각 원소의 소수자리를 내림
# rint 각 원소의 소수자리를 반올림. dtype 은 유지
# modf 각 원소의 몫과 나머지를 각각의 배열로 반환
# isnan 각 원소가 숫자가 아닌지를 나타내는 불리언 배열 반환
# isfinite, isinf 각각 배열의 각 원소가 유한한지 무한한지 나타내는 불리언 배열 반환
# cos, cosh, sin, sinh, 일반 삼각함수와 쌍곡삼각함수
# tan, tanh
# arccos, arccosh, arcsin, 역삼각함수
# arcsinh, arctan, arctanh
# logical_not 각 원소의 논리 부정(not)값을 계산. ~arr 과 동일

# 이항 유니버설 함수
# # 함수 설명
# # add 두 배열에서 같은 위치의 원소끼리 더하기
# # subtract 첫번째 배열의 원소 - 두번째 배열의 원소
# # multiply 배열의 원소끼리 곱하기
# # divide, floor_divide 첫번째 배열의 원소를 두 번째 배열의 원소로 나눈다.
# # floor_dived 는 몫만 취한다.
# # power 첫번째 배열의 원소를 두 번째 배열의 원소만큼 제곱한다.
# # maximum, fmax 두 원소 중 큰 값 반환. fmax 는 NaN 무시
# # minimum, fmin 두 원소 중 작은 값을 반환. fmin 은 NaN 을 무시
# # mod 나머지
# # copysign 첫 번째 배열의 원소의 기호를 두번째 배열의 원소의 기호로
# 바꾼다.
# # greater, greater_equal, less, 두 원소 간의 >, >=, <, <=, ==, != 비교 연산 결과를
# 불리언 배열로 반환
# # less_equal, equal, not_equal,
# # logical_and, logical_or, 각각 두 원소간의 &, |, ^ 논리 연산 결과를 반환
# # logical_xor


## 4.3 Array-Oriented Programming with Arrays
# 벡터화: 배열 연산을 사용해서 반복문을 명시적으로 제거하는 기법
# 일반적으로 벡터화된 배열에 대한 산술연산은 순수 파이썬 연산에 비해 2~3 배에서 많게는 수십, 수백배까지 빠르다.
# 값이 놓여 있는 그리드에서 sqrt(x^2 + y^2)를 계산
# np.meshgrid 함수는 두개의 1 차원 배열을 받아서 가능한 모든 (x, y)짝을 만들 수 있는 2 차원배열 두개를 반환
points = np.arange(-5, 5, 0.01) # 1000 equally spaced points
xs, ys = np.meshgrid(points, points)
ys
xs
# 그리드 상의 두 포인트로 간단하게 계산 적용 가능
z = np.sqrt(xs ** 2 + ys ** 2)
z
# matplotlib 이용 2 차원 배열 시각화
import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.draw() # re-draw. interactive mode 에서 data 나 format 을 바꾸어야 할 때 사용
plt.close('all')



# 4.3.1 Expressing Conditional Logic as Array Operations
# np.where 함수: x if 조건 else y 같은 삼항식의 벡터화된 버전
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
# cond 의 값이 True 일때는 xarr 값, False 일때는 yarr 값
result = [(x if c else y)
 for x, y, c in zip(xarr, yarr, cond)]
result
# 이 방법은 상대적으로 느리고 다차원 배열에서는 사용할 수 없다.
# np.where 이용.
result = np.where(cond, xarr, yarr)
result
# np.where 의 두 번째와 세 번째 인자는 배열이 아니어도 상관없음.
# 둘 중 하나 혹은 둘 다 스칼라값이어도 동작
# where 은 다른 배열에 기반한 새로운 배열을 생성
arr = np.random.randn(4, 4)
arr
arr > 0
np.where(arr > 0, 2, -2)
# arr 의 모든 양수를 2 로 바꿈.
np.where(arr > 0, 2, arr) # set only positive values to 2
arr


# 4.3.2 Mathematical and Statistical Methods
# 임의의 정규 분포 데이터 생성 및 집계
arr = np.random.randn(5, 4)
arr
arr.mean()
np.mean(arr)
arr.sum()
# axis 인자를 받아서 해당 axis 에 대한 통계를 계산하고 한 차수 낮은 배열 반환
arr.mean(axis=1)
arr.sum(axis=0)
# where arr.sum(axis=0)은 row 의 합을 구하라는 의미, arr.mean(axis=1)은 모든 컬럼에서 평균을 구하라는 의미
# cumsum 과 cumprod 메서드는 중간 계산값을 담고 있는 배열 반환
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr
arr.cumsum()
# 다차원 배열에서 cumsum 함수는 같은 크기의 배열 반환
# 축을 지정하여 부분적으로 계산하면 낮은 차수의 슬라이스를 반환
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr
arr.cumsum(axis=0)
arr.cumprod(axis=1)

# 기본 배열 통계 메서드
# # 메서드 설명
# # sum 모든 원소의 합
# # mean 산술 평균. 크기가 0 인 배열에 대한 mean 결과는 NaN
# # std, var 표준편차, 분산
# # min, max 최소값, 최대값
# # argmin, argmax 최소 원소의 색인값, 최대 원소의 색인값
# # cumsum 누적합
# # cumprod 누적곱



# 4.3.3 Methods for Boolean Arrays
# sum 메서드를 실행하면 불리언 배열에서 True 인 원소의 개수 셀 수 있다.
arr = np.random.randn(100)
(arr > 0).sum() # Number of positive values
# any 메서드는 하나 이상의 값이 True 인지 검사
# all 메서드는 모든 원소가 True 인지 검사
bools = np.array([False, False, True, False])
bools.any()
bools.all()



# 4.3.4 Sorting
# sort 메서드 이용
arr = np.random.randn(6)
arr
arr.sort()
arr
# * 내림차순
arr_desc = arr[::-1]
arr_desc
# 다차원 배열의 정렬은 sort 메서드에 넘긴 축의 값에 따라 1 차원 부분([ ]안)을 정렬
arr = np.random.randn(5, 3)
arr
arr.sort(1)
arr
# * [ ] 안 내림차순 정렬
arr_innerdesc = arr[:,::-1]
arr_innerdesc
# np.sort 메서드는 배열을 직접 변경하지 않고 정렬된 결과를 가지고 있는 복사본을 반환
large_arr = np.random.randn(10)
large_arr
large_arr.sort()
large_arr[int(0.05 * len(large_arr))] # 5% quantile
# 표 형식의 데이터를 하나 이상의 열로 정렬하는 것 같은 정렬과 관련된 여러가지 데이터 처리 내용은
# pandas 에서 다룬다.



# 4.3.5 Unique and Other Set Logic
# NumPy 는 1 차원 ndarray 를 위한 기본적인 집합 연산 제공
# np.unique : 배열 내 중복된 원소를 제거하고 남은 원소를 정렬된 형태로 반환
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)
# np.unique 를 순수 파이썬으로 구현
sorted(set(names))
# np.in1d 함수: 두 개의 배열을 인자로 받아서 첫 번째 배열의 원소가 두 번째 배열의 원소를 포함하는지 나타내는 불리언 배열을 반환
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

# 표 4-6 배열 집합 연산
#
# 메서드 설명
# unique(x) 배열 x 에서 중복된 원소 제거 후 정렬하여 반환
# intersect1d(x,y) 공통적인 원소를 정렬하여 반환
# union1d(x) 합집합 반환
# in1d(x,y) x 의 원소가 y 의 원소에 포함되는지를 불리언 배열로 반환
# setdiff1d(x,y) 차집합 반환
# setxor1d(x,y) 한 배열에는 포함되지만 두 배열 모두에는 포함되지 않는 원소들의 집합인 대칭차집합 반환


## 4.4 File Input and Output with Arrays
# 텍스트나 표 형식의 데이터는 pandas 사용

# np.save, np.load: 배열 데이터를 디스크에 저장하고 불러오기 위한 함수
arr = np.arange(10)
np.save('some_array', arr)

# 저장되는 파일 경로가 .npy 로 끝나지 않으면 자동적으로 확장자가 추가된다.
# 이렇게 저장된 배열은 np.load 를 이용하여 불러올 수 있다.
np.load('some_array.npy')

# np.savez 함수: 여러 개의 배열을 압축된 형식으로 저장
np.savez('array_archive.npz', a=arr, b=arr)

# npz 파일을 불러올 때는 각각의 배열을 필요할 때 불러올 수 있도록 사전 형식의 객체에 저장
arch = np.load('array_archive.npz')
arch['a']

# np.savez_compressed : 압축이 잘되는 형식의 데이터인 경우 사용
np.savez_compressed('arrays_compressed.npz', a=arr, b=arr)


 4.5 Linear Algebra
# 2 차원 배열을 * 연산자로 곱하면 행렬 곱셈이 아니라 원소의 곱을 계산
# 행렬의 곱셈은 dot 함수 이용
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x
y

# x.dot(y) == np.dot(x, y)
x.dot(y)
np.dot(x, y)

# 2차원 배열과 곱셈이 가능한 크기의 1차원 배열 간의 행력 곱셈의 결과는 1차원 배열
np.dot(x, np.ones(3))

# @기호 : 행렬 곱셈을 수행하는 연사자 (파이썬 3.5부터 사용가능)
x @ np.ones(3)

# np.linalg: 행렬의 분할과 역행렬, 행렬식과 같은 것 포함.
from numpy.linalg import inv, qr
X = np.random.randn(5, 5)
X
mat = X.T.dot(X)
mat
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)
q
r

# X.T.dot(X): X 의 전치행렬과 X 의 곱

# 표 4-7 자주 사용하는 np.linalg 함수
# 함수 설명
# diag 정사각 행렬의 대각/비대각 원소를 1 차원 배열로 반환,
# 또는 1 차원 배열을 대각선 원소로하고, 나머지는 0 으로 채운 단위행렬
# dot 행렬 곱셈
# trace 행렬의 대각선 원소의 합 계산
# det determinant
# eig 고유값과 고유벡터
# inv 역행렬 계산
# pinv 무어-펜로즈 유사역원 역행렬
# qr QR 분해 계산
# svd 특이값 분해(SVD) 계산
# solve A 가 정사각 행렬일때 Ax = b 를 만족하는 x 구함
# lstsq Ax = b 를 만족하는 최소제곱해를 구함



# 4.6 Pseudorandom Number Generation
# 표준정규분포로부터 4x4 크기의 표본 생성
samples = np.random.normal(size=(4, 4))
samples
# 파이썬 내장 모듈보다 수십 배 이상 빠르다.
from random import normalvariate
N = 1000000
from datetime import datetime
start = datetime.now()
samples = [normalvariate(0, 1) for _ in range(N)]
np.random.normal(size=N)
print(datetime.now() - start)
# 유사난수: 난수 생성기의 시드값에 따라 정해진 난수를 알고리즘으로 생성하기 때문에 유사난수로 부름.
# NumPy 난수 생성기의 시드값은 np.random.seed 를 이용해서 변경 가능
np.random.seed(1234)
# np.random.RandomState 를 이용하여 다른 난수 생성기로부터 격리된 난수 생성기를 만들 수 있음.
rng = np.random.RandomState(1234)
rng.randn(10)
# 차이점
# np.random.seed: NumPy 에 존재하는 random generator 에 직접 접근하여 난수 생성
# np.random.RandomState: 난수 생성기라는 object 를 만들어서 접근
np.random.seed(1234)
np.random.uniform(0, 10, 5)
np.random.rand(3,3)

rng2 = np.random.RandomState(1234)
rng2.uniform(0, 10, 5)
np.random.rand(3,3)


# 표 4-8
# 함수 설명
# seed 난수 생성기의 시드 지정
# permutation 순서를 임의로 바꾸거나 임의의 순열을 반환
# shuffle 리스트나 배열의 순서를 뒤섞는다.
# rand 균등분포에서 표본 추출
# randint 주어진 최소/최대 범위 안에서 임의의 난수 추출
# randn 표준편차가 1 이고 평균값이 0 인 정규분포에서 표본 추출
# binomial 이항분포에서 표본 추출
# normal 정규분포에서 표본 추출
# beta 베타분포에서 표본 추출
# chisquare 카이제곱분포에서 표본 추출
# gamma 감마분포에서 표본 추출
# uniform 균등[0,1)분포에서 표본 추출


## 4.7 Example: Random Walks
# 계단 오르내리기 예제
# 순수 파이썬 내 내장 random 모듈 사용하여 계단 오르내리기를 1,000 번 수행하는 코드:
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
 step = 1 if random.randint(0, 1) else -1
 position += step
 walk.append(position)
plt.figure()

# 처음 100 회 계단 오르내리기를 그래프화
plt.plot(walk[:100])
# 1,000 번 수행
np.random.seed(12345)
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
# 계단을 오르내린 위치의 최소값과 최대값
walk.min()
walk.max()
# 최초의 10 혹은 -10 인 시점
(np.abs(walk) >= 10).argmax()



# 4.7.1 Simulating Many Random Walks at Once
# np.random 함수에 크기가 2 인 튜플을 넘기면 2 차원 배열이 생성
# 각 컬럼에서 누적합을 구해서 5,000 회의 시뮬레이션을 한번에 처리
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks
# 모든 시뮬레이션에 대해 최대값과 최소값
walks.max()
walks.min()
# 누적합이 30 또는 -30 에 도달하는 최소시점 계산
hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum() # Number that hit 30 or -30
# 처음 위치에서 30 칸 이상 멀어지는 최소 횟수:
# 컬럼 선택하고 절대값이 30 을 넘는 경우에 대해 축 1 의 argmax 값
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()
# normal 함수 이용
steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))
steps