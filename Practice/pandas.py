# Python for Data Analysis 2nd
# ch5 Pandas
### Getting Started with pandas
# pandas 라이브러리: 고수준의 자료구조와 빠르고 쉽게 사용할 수 있는 데이터 분석 도구 포함
# NumPy, SciPy: 산술계산 도구
# statsmodels, scikit-learn: 분석 라이브러리
# matplotlib: 시각화 도구
# pandas 는 for 문을 사용하지 않고 데이터를 처리한다거나 배열 기반의 함수를 제공하는 등
# NumPy 의 배열 기반 계산 스타일을 많이 차용
# pandas 와 NumPy 의 차이점:
# pandas: 표 형식의 데이터나 다양한 형태의 데이터를 다루는 데 초점을 맞춰 설계
# NumPy: 단일 산술 배열 데이터를 다루는데 특화

pip install --upgrade pandas

# pandas 의 import 컨벤션
import pandas as pd

# Series 와 DataFrame 은 로컬 네임스페이스로 인포트
from pandas import Series, DataFrame

# 환경설정
import numpy as np
np.random.seed(12345)
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)
pd.options.display.max_rows = PREVIOUS_MAX_ROWS



### 5.1 Introduction to pandas Data Structures
# 5.1.1 Series
# Series 는 1 차원 배열 같은 자료구조(어떤 NumPy 자료형이라도 담을 수 있다.)
# 색인(Index)이란 배열의 데이터와 연관된 이름을 가지고 있다.
# 간단한 Series 객체를 배열 데이터로부터 생성
obj = pd.Series([4, 7, -5, 3])
obj
# Series 객체의 문자열 표현에서 왼쪽은 색인, 오를쪽은 해당 색인의 값을 보여준다.
# 데이터의 색인이 지정되지 않았다면 기본 색인인 정수 0 부터 N-1(N 은 데이터의 길이)까지의 숫자가 표시된다.
obj.values
obj.index # like range(4)
# Series 객체 생성
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2.index
# 단일 값을 선택하거나 여러 값을 선택할 때 색인으로 라벨(Label)을 사용할 수 있다.
obj2['a']
obj2['d'] = 6
obj2[['c', 'a', 'd']]
# 여기서 ['c', 'a', 'd']는 (정수가 아니라 문자열이 포함되어 있지만) 색인의 배열로 해석된다.
# 불리언 배열을 사용해서 값을 걸러내거나 산술 곱셈을 수행하거나 또는 수학 함수를 적용하는 등
# NumPy 배열 연산을 수행해도 색인-값 연결이 유지된다.
obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)
# Series: 고정 길이의 정렬된 dict 라고 생각
# Series 는 색인값에 데이터값을 매핑하고 있으므로 파이썬의 사전형(dict)과 비슷
# Series 객체는 파이썬의 사전형을 인자로 받아야 하는 많은 함수에서 사전형을 대체하여 사용 가능
'b' in obj2
'e' in obj2

# 파이썬 사전형(dict)에 데이터를 저장해야 한다면 파이썬 사전 객체로부터 Series 객체를 생성할 수있다.
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)
obj3
# 사전 객체만 가지고 Series 객체를 생성하면 생성된 Series 객체의 색인에는 사전의 키값이 순서대로 들어간다.
# 색인을 직접 지정하고 싶다면 원하는 순서대로 색인을 직접 넘겨줄 수도 있다.
states = ['California', 'Ohio', '', 'Texas']
pd.Series(states)
obj4 = pd.Series(sdata, index=states)
obj4
# sdata 에 3 개의 값만 있고 'California'에 대한 값은 없기 때문에
# 값은 NaN 으로 표시되고, pandas 에서는 누락된 값 또는 NA 값으로 취급
# 'Utah'는 states 에 포함되어 있지 않으므로 실행 결과에서 빠짐
# isnull 과 notnull 함수는 누락된 데이터를 찾을 때 사용
pd.isnull(obj4)
pd.notnull(obj4)
# 이 메서드는 Series 의 인스턴스 메서드로도 존재
obj4.isnull()
# 산연산에서 색인과 라벨로 자동 정렬
obj3
obj4
obj3 + obj4
# join 연산과 유사
# Series 객체와 Series 색인은 모두 name 속성을 가진다.
obj4.name = 'population'
obj4.index.name = 'state'
obj4
# Series 의 색인을 대입하여 변경 가능
obj
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj.index.name = 'Human'
obj


# 5.1.2 DataFrame
# DataFrame 은 표 같은 스프레드시크 형식의 자료구조
# 여러개의 컬럼을 가지는데 서로 다른 종류의 값(숫자, 문자열, 불리언 등)을 담을 수 있다.
# DataFrame 은 로우와 컬럼에 대한 색인을 가지고 있다.
# 색인의 모양이 같은 Series 객체를 담고 있는 파이썬 사전으로 생각하면 편하다.
# 내부적으로 데이터는 리스트나 사전 또는 1 차원 배열을 담고 있는 다른 컬렉션이 아니라
# 하나 이상의 2 차원 배열에 저장된다.
# 물리적으로 DataFrame 은 2 차원이지만 계층적 색인을 이용해서 좀 더 고차원의 데이터를 표현할 수 있다.
# DataFrame 객체 생성
# 같은 길이의 리스트에 담긴 사전을 이용하거나 NumPy 배열을 이용
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
 'year': [2000, 2001, 2002, 2001, 2002, 2003],
 'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
# 만들어진 DataFrame 의 색인은 Series 와 같은 방식으로 자동으로 대입되며 컬럼은 정렬되어 저장된다.
frame
# head 메서드를 이용하여 처음 5 개의 로우만 출력 가능
frame.head()
# 원하는 순서대로 columns 를 지정하면 원하는 순서를 가진 DataFrame 객체 생성
pd.DataFrame(data, columns=['year', 'state', 'pop'])
# Series 와 동일하게 사전에 없는 값을 넘기면 결측치로 저장된다.
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
 index=['one', 'two', 'three', 'four',
 'five', 'six'])
frame2
frame2.columns
# DataFrame 의 컬럼은 Series 처럼 사전 형식의 표기법으로 접근하거나 속성 형식으로 접근할 수 있다.
frame2['state']
frame2.year
# * frame2[column] 형태로 사용하는 것은 어떤 컬럼이든 가능.
# 하지만 frame2.column 형태로 사용하는 것은 사용가능한 변수이름 형식일때만 작동
# 반환된 Series 객체가 DataFrame 과 같은 색인을 가지면 알맞은 값으로 name 속성이 채워진다.
# 로우는 위치나 loc 속성을 이용하여 이름을 통해 접근할 수 있다.
frame2.loc['three']
# 컬럼은 대입이 가능하다.
# 예를 들어 현재 비어 있는 'debt'컬럼에 스칼라값이나 배열의 값을 대입할 수 있다.
frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(6.)
frame2

# 리스트나 배열을 컬럼에 대입할 때는 대입하려는 값의 길이가 DataFrame 의 크기와 동일해야 한다.
# Series 를 대입하면 DataFrame 의 색인에 따라 값이 대입되며 존재하지 않는 색인에는 결측치가 대입된다.
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
# 존재하지 않는 컬럼을 대입하면 새로운 컬럼을 생성
frame2['eastern'] = frame2.state == 'Ohio'
frame2
# del 예약어를 사용하여 컬럼 삭제 가능
del frame2['eastern']
frame2.columns
# DataFrame 의 색인을 이용해서 얻은 컬럼은 내부 데이터에 대한 뷰(view)이며 복사가 이루어지지 않는다.
# 이렇게 얻은 Series 객체에 대한 변경은 실제 DataFrame 에 반영된다.
# 복사본이 필요할 때는 Series 의 copy 메서드를 이용한다.
# 중첩된 사전을 이용하여 데이터 생성 가능
# 중첩된 사전
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
# 중첩된 사전을 DataFrame 에 넘기면 바깥에 있는 사전의 키는 컬럼이 되고 안에 있는 키는 로우가 된다.
frame3 = pd.DataFrame(pop)
frame3
# 데이터의 전치(transpose) 가능
frame3.T
# 중첩된 사전을 이용하여 DataFrame 을 생성할 때 안쪽에 있는 사전값은 키값별로 조합되어 결과의 색인이 되지만
# 색인을 직접 지정하면 지정된 색인으로 DataFrame 을 생성
pd.DataFrame(pop, index=[2001, 2002, 2003])
# Series 객체를 담고 있는 사전 데이터도 같은 방식으로 취급된다.
pdata = {'Ohio': frame3['Ohio'][:-1],
 'Nevada': frame3['Nevada'][:2]}
pd.DataFrame(pdata)
# 데이터프레임의 색인(index)와 컬럼(columns)에 name 속성을 지정했다면 함께 출력된다.
frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3
# Series 와 유사하게 values 속성은 DataFrame 에 저장된 데이터를 2 차원 배열로 반환
frame3.values
# DataFrame 의 컬럼이 서로 다른 dtype 을 가지고 있다면
# 모든 컬럼을 수용하기 위해 그 컬럼의 배열의 dtype 이 선택된다.
frame2.values

# DataFrame 생성을 위한 입력 데이터의 종류
# 형 설명
# 2 차원 ndarray 데이터를 담고 있는 행렬, 선택적으로 행(로우)과 열(컬럼)이 이름을 전달할 수 있다.
# 배열, 리스트, 튜플의 사전 사전의 모든 항목은 같은 길이를 가져야 하며, 각 항목의 내용이 DataFrame 의 컬럼이 된다.
# NumPy 의 구조화 배열 배열의 사전과 같은 방식으로 취급된다
# Series 의 사전 Series 의 각 값이 컬럼이 된다. 명시적으로 색인을 넘겨주지 않으면
# 각 Series 의 색인이 하나로 합쳐져서 로우의 색인이 된다.
# 사전의 사전 내부에 있는 사전이 컬럼이 된다. 키값은 'Series 의 사전'과 마찬가지로 합쳐져서 로우의 색인이 된다.
# 사전이나 Series 의 리스트 리스트의 각 항목이 DataFrame 의 로우가 된다. 합쳐진 사전의 키값이나 Series 의 색인이 DataFrame 의 컬럼 이름이 된다.
# 리스트나 튜플의 리스트 '2 차원 ndarray'의 경우와 같은 방식으로 취급된다.
# 다른 DataFrame 색인을 따로 지정하지 않으면 DataFrame 의 색인이 그대로 사용된다.
# NumPy MaskedArray '2 차원 ndarray'의 경우와 같은 방식으로 취급되지만 마스크값은 반환되는 DataFrame 에서 NA 값이 된다.



# 5.1.3 Index Objects
# pandas 의 색인 객체는 표 형식의 데이터에서 각 로우와 컬럼에 대한 이름과 다른 메타데이터(축의 이름 등)를 저장하는 객체
# Series 나 DataFrame 객체를 생성할 때 사용되는 배열이나 다른 순차적인 이름은 내부적으로 색인으로 변환된다.
obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index
index[1:]
# 색인 객체는 변경이 불가능
index[1] = 'd' # TypeError 발생
# 자료구조 사이에서 안전하게 공유될 수 있다.
labels = pd.Index(np.arange(3))
labels
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2
obj2.index is labels
# 배열과 유사하게 Index 객체도 고정 크기로 동작
frame3
frame3.columns
'Ohio' in frame3.columns
2003 in frame3.index
# 파이썬의 집합과는 달리 pandas 의 인덱스는 중복되는 값을 허용
dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
dup_labels
# 중복되는 값으로 선택을 하면 해당 값을 가진 모든 항목이 선택된다.

# 색인 메서드와 속성
# 메서드 설명
# append 추가적인 색인 객체를 덧붙여 새로운 색인을 반환
# difference 색인의 차집합을 반환
# intersection 색인의 교집합을 반환
# union 색인의 합집합을 반환
# isin 색인이 넘겨받은 색인에 존재하는지 알려주는 불리언 배열을 반환
# delete i 위치의 색인이 삭제된 새로운 색인을 반환
# drop 넘겨받은 값이 삭제된 새로운 색인을 반환
# insert i 위치의 색인이 추가된 새로운 색인을 반환
# is_monotonic 색인이 단조성을 가진다면 True 반환
# is_unique 중복되는 색인이 없다면 True 반환
# unique 색인에서 중복되는 요소를 제거하고 유일한 값만 반환



## 5.2 Essential Functionality
# Series 나 DataFrame 에 저장된 데이터를 다루는 기본적인 방법 설명

# 5.2.1 Reindexing
# reindex: 새로운 색인에 맞도록 객체를 새로 생성
# 예제
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj
# Series 객체에 대해 reindex 를 호출하면 데이터를 새로운 색인에 맞게 재배열하고,
# 존재하지 않는 색인 값이 있다면 NaN 을 추가한다.
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2
# 시계열 데이터를 재색인할 때 값을 보간하거나 채워 넣어야 할 경우
# method 옵션을 이용하여 실행
# ffill 메서드를 이용하여 누락된 값을 직전의 값으로 채워 넣을 수 있다.
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3
obj3.reindex(range(6), method='ffill')
# DataFrame 에 대한 reindex 는 로우(색인), 컬럼 또는 둘다 변경 가능
# 순서만 전달하면 로우가 재색인된다.
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
 index=['a', 'c', 'd'],
 columns=['Ohio', 'Texas', 'California'])
frame
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2
# 컬럼은 columns 예약어를 사용하여 재색인
states = ['Texas', 'Utah', 'California']
qw = frame2.reindex(columns=states)
# 재색인은 loc 를 이용하여 라벨로 색인하면 좀 더 간결하게 할 수 있다.
qw.loc[['a', 'b', 'c', 'd'], states]
# 표 5-3 재색인 함수 인자
# 인자 설명
# index 색인으로 사용할 새로운 순서. index 는 복사가 이루어지지 않고 그대로 사용된다
# method 채움메서드 ffill 은 직전 값을 채워 넣고 bfill 은 다음 값을 채워 넣는다
# fill_value 재색인 과정 중에 새롭게 나타나는 비어 있는 데이터를 채우기 위한 값
# limit 전/후 부간 시에 사용할 최대 캡 크기(채워넣을 원소의 수)
# tolerance 전/후 보간 시에 사용할 최대 갭 크기(값의 차이)
# level MultiIndex 의 단계(level)에 단순 색인을 맞춘다.
# 그렇지 않으면 MultiIndex 의 하위집합에 맞춘다.
# copy True 인 경우 새로운 색인이 이전 색인과 동일하더라도 데이터를 복사한다.
# False 인 경우 새로운 색인이 이전 색인과 동일할 경우 복사하지 않는다.



# 5.2.2 Dropping Entries from an Axis
# 색인 배열, 또는 삭제하려는 로우나 컬럼이 제외된 리스트를 가지고 있다면
# 로우나 컬럼을 쉽게 삭제 가능.
# 이 방법은 데이터의 모양을 변경하는 작업이 필요
# drop 메서드를 사용하면 선택한 값들이 삭제된 새로운 객체를 얻을 수 있다.
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])
# DataFrame 에서는 로우와 컬럼 모두에서 값을 삭제할 수 있다.
# 예제
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
 index=['Ohio', 'Colorado', 'Utah', 'New York'],
 columns=['one', 'two', 'three', 'four'])
data
# drop 함수에 인자로 로우 이름을 지정하면 해당 로우 (axis())의 값을 모두 삭제
data.drop(['Colorado', 'Ohio'])
# 컬럼 값을 삭제할 때는 axis=1 또는 axis='columns'를 인자로 넘겨주면 된다.
data.drop('two', axis=1)
data.drop(['two', 'four'], axis='columns')
# drop()함수 처럼 Series 나 DataFrame 의 크기 또는 형태를 변경하는 함수는
# 새로운 객체를 반환하는 대신 원본 객체를 변경한다.
obj.drop('c', inplace=True)
obj
# inplace 옵션을 사용하는 경우 버려지는 값을 모두 삭제하므로 주의!


# 5.2.3 Indexing, Selection, and Filtering
# Series 의 색인(obj[...])은 NumPy 배열의 색인과 유사하게 동작하지만 정수가 아니어도 된다는 점이 다르다.
# 예제
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj
obj['b']
obj[1]
obj[2:4]
obj[['b', 'a', 'd']]
obj[[1, 3]]
obj[obj < 2]
# 라벨 이름으로 슬라이싱하면 시작점과 끝점을 포함한다는 것이 일반 파이썬에서의 슬라이싱과 다른 점.
obj['b':'c']
# 슬라이싱 문법으로 선택된 영역에 값을 대입하는 것은 생각하는대로 동작
obj['b':'c'] = 5
obj
# 색인으로 DataFrame 에서 하나 이상의 컬럼 값을 가져올 수 있음.
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
 index=['Ohio', 'Colorado', 'Utah', 'New York'],
 columns=['one', 'two', 'three', 'four'])
data
data['two']
data[['three', 'one']]
# 슬라이싱으로 로우를 선택하거나 불리언 배열로 로우를 선택 가능
data[:2]
data[data['three'] > 5]
# 다른 방법으로 스칼라 비교를 이용하여 생성된 불리언 DataFrame 을 사용하여 값을 선택
data < 5
data[data < 5] = 0
data
# Selection with loc and iloc
# loc & iloc: DataFrame 의 로우에 대해 라벨로 색인하는 방법으로 특수 색인 필드
# 축의 라벨을 사용하여 DataFrame 의 로우와 컬럼을 선택 가능.
# 축 이름을 선택할 때는 loc 를, 정수색인으로 선택할 때는 iloc 사용
data.loc['Colorado', ['two', 'three']]
data.iloc[2, [3, 0, 1]]
data.iloc[2]
data.iloc[[1, 2], [3, 0, 1]]
# loc & iloc 함수는 슬라이스 지원 및 단일 라벨이나 라벨 리스트 지원
data.loc[:'Utah', 'two']
data.iloc[:, :3][data.three > 5]
# 표 5-4 DataFrame 의 값 선택하기
# 방식 설명
# df[val] DataFrame 에서 하나의 컬럼 또는 여러 컬럼 선택
# df.loc[val] DataFrame 에서 라벨값으로 로우의 부분집합을 선택
# df.loc[:, val] DataFrame 에서 라벨값으로 컬럼의 부분집합을 선택
# df.loc[val1, val2] DataFrame 에서 라벨값으로 로우와 컬럼의 부분집합을 선택
# df.loc[where] DataFrame 에서 정수 색인으로 로우의 부분집합을 선택
# df.iloc[:, where] DataFrame 에서 정수 색인으로 컬럼의 부분집합을 선택
# df.iloc[where_i, label_i] 로우와 컬럼의 라벨로 단일 값 선택
# df.iat[i, j] 로우와 컬럼의 정수 색인으로 단일 값 선택
# reindex 메서드 하나 이상의 축을 새로운 색인으로 맞춘다.
# get_value, set_value 메서드 로우와 컬럼 이름으로 DataFrame 값 선택


# 5.2.4 Integer Indexes
# 정수 색인으로 pandas 객체를 다루다보면 리스트나 튜플 같은 파이썬 내장 자료구조에서 색인을 다루는 방법과의 차이점 때문에 실수한다

# 예)
## ser = pd.Series(np.arange(3.)); ser; ser[-1]
# check!
# pandas 는 라벨 색인을 찾는데 실패하므로 정수 색인으로 값을 찾는다.
# 라벨 색인이 0, 1, 2 를 포함하는 경우 사용자가 라벨 색인으로 선택하려는 것인지 정수
# 색인으로 선택하려는 것인지 추측하기 어렵다.
ser = pd.Series(np.arange(3.))
ser
# 정수 기반의 색인을 사용하지 않는 경우 이런 모호함은 사라짐.
ser2 = pd.Series(np.arange(3.), index=['a', 'b', 'c'])
ser2[-1]
# 일관성 유지를 위해 정수값을 담고 있는 축 색인이 있다면 우선적으로 라벨을 먼저 찾아보도록 구현되어 있음.
# 라벨에 대해서는 loc 을 사용하고 정수 색인에 대해서는 iloc 을 사용하자.
ser[:1]
ser.loc[:1]
ser.iloc[:1]

import pandas as np

# 5.2.5 Arithmetic and Data Alignment
# pandas 에서 가장 중요한 기능 중 하나는 다른 색인을 가지고 있는 객체 간의 산술 연산이다.
# 객체를 더할 때 짝이 맞지 않는 색인이 있다면 결과에 두 색인이 통합된다.
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1
s2
# 외부조인과 유사하게 동작
# 서로 겹치는 색인이 없는 경우 데이터는 NA 값
s1 + s2
# DataFrame 의 경우 정렬은 로우와 컬럼 모두에 적용됨.
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
 index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
 index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2
df1 + df2
# 공통되는 컬럼 라벨이나 로우 라벨이 없는 DataFrame 을 더하면 결과에 아무것도 안 나타남.
df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'B': [3, 4]})
df1
df2
df1 - df2
# Arithmetic methods with fill values
# 서로 다른 색인을 가지는 객체 간의 산술연산에서 존재하지 않는 축의 값을 특수한 값(예, 0)으로 지정 시
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
 columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
 columns=list('abcde'))
df2.loc[1, 'b'] = np.nan
df1
df2
# 겹치지 않는 부분은 NA 값
df1 + df2

# df1 에 add 메서드 사용
df1.add(df2, fill_value=0)
# 메서드는 r 로 시작하는 짝꿍메서드를 가진다.
1 / df1
df1.rdiv(1)
# 재색인
df1.reindex(columns=df2.columns, fill_value=0)
# 산술 연산 메서드
# 메서드 설명
# add, radd (+)을 위한 메서드
# sub, rsub (-)을 위한 메서드
# div, rdiv (/)을 위한 메서드
# floordiv, rfloordiv 소수점 내림(//) 연산을 위한 메서드
# mul, rmul (*)을 위한 메서드
# pow, rpow (**)을 위한 메서드
# Operations between DataFrame and Series

# 2 차원 배열과 그 배열의 한 로우의 차이에 대해 생각할 수 있는 예제
arr = np.arange(12.).reshape((3, 4))
arr
arr[0]
# arr - arr[0]은 각 로우별 한 번씩 수행 --> '브로드캐스팅'
arr - arr[0]
# DataFrame 와 series 간의 연산은 이와 유사
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
 columns=list('bde'),
 index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.iloc[0]
frame
series
# DataFrame 과 Series 간의 산술연산은 Series 의 색인을 DataFrame 의 컬럼에 맞추고 아래 로우로 전파
frame - series
# 색인값을 DataFrame 의 컬럼이나 Series 의 색인에서 찾을 수 없다면 그 객체는 형식을 맞추기 위해 재색인된다.
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
series2
frame + series2
# 각 로우에 대해 연산을 수행하고 싶다면 산술 연산 메서드를 사용
series3 = frame['d']
frame
series3
frame.sub(series3, axis='index') # 빼기


# 5.2.7 Sorting and Ranking
# 로우나 컬럼의 색인을 알파벳순으로 정렬을 하려면 sort_index 메서드 사용
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()
# DataFrame 은 로우나 컬럼 중 하나의 축을 기준으로 정렬 가능
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
 index=['three', 'one'],
 columns=['d', 'a', 'b', 'c'])
frame
frame.sort_index()
frame.sort_index(axis=0)
frame.sort_index(axis=1)
# default 는 오름차순. 내림차순으로 정렬 가능.
frame.sort_index(axis=1, ascending=False)
# Series 객체를 값에 따라 정렬하고 싶다면 sort_values 메서드 사용
obj = pd.Series([4, 7, -3, 2])
obj.sort_values()
# 정렬할 때 비어 있는 값은 Series 객체에서 가장 마지막에 위치
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()
# DataFrame 에서 하나 이상의 컬럼에 있는 값으로 정렬을 하는 경우
# sort_value 함수의 by 옵션에 하나 이상의 컬럼 이름을 넘기면 된다.
frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame
frame.sort_values(by='b')
# 여러 개의 컬럼을 정렬하려면 컬럼 이름이 담긴 리스트 사용
frame.sort_values(by=['a', 'b'])
# 순위는 정렬과 유사. 1 부터 배열의 유효한 데이터 갯수까지 순서를 매긴다.
# Series 와 DataFrame 의 rank 메서드는 동점인 경우 평균 순위 표시
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj
obj.rank()
# 데이터 상에서 나타나는 순서에 따라 순위를 매길 수 도 있다.
obj.rank(method='first')
# 0 번째와 2 번째 항목에 대해 평균 순위인 6.5 대신 출현한 순서대로 6 가 7 적용
# 내림차순으로 순위 정렬
# Assign tie values the maximum rank in the group
obj.rank(ascending=False, method='max')
# DataFrame 에서는 로우나 컬럼에 대해 순위 결정 가능
frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
 'c': [-2, 5, 8, -2.5]})
frame
frame.rank(axis='columns')
# 5-6
# 메서드 설명
# 'average' 기본값은 값을 가지는 항목들의 펴윤값을 순위로 삼는다.
# 'min' 같은 값을 가지는 그룹을 낮은 순위로 매긴다.
# 'max' 같은 값을 가지는 그룹을 높은 순위로 매긴다.
# 'first' 데이터 내의 위치에 따라 순위를 매긴다.
# 'dense' method ='min'과 같지만 같은 그룹 내에서 모두 같은 순위를 적용하지 않고 1 씩 증가


# 5.2.8 Axis Indexes with Duplicate Labels 중복색인
# pandas 의 많은 함수(예, reindex)에서 색인값은 유일해야 하지만 의무적이지 않다.
# 중복된 색인값을 가지는 Series 객체 예제
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj
# 색인이 유일한 값인지 확인
obj.index.is_unique
# 중복되는 색인값이 있는 경우 색인을 이용하여 데이터에 접근하면 하나의 Series 객체 반환
obj['a']
obj['c']
# DataFrame 에서 로우 선택하는 것도 동일
df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df
df.loc['b']


## 5.3 Summarizing and Computing Descriptive Statistics
# pandas 객체는 일반적인 수학 메서드와 통계 메서드를 가지고 있음.
# pandas 의 메서드는 처음부터 누락된 데이터를 제외하도록 설계되어 있음.
# 예시
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
 [np.nan, np.nan], [0.75, -1.3]],
 index=['a', 'b', 'c', 'd'],
 columns=['one', 'two'])
df
# DataFrame 의 sum 메서드를 호출하면 각 컬럼의 합을 담은 Series 반환
df.sum()
# option 사용
df.sum(axis='columns')
# 전체 로우나 컬럼의 값이 NA 가 아니라면 NA 값을 제외하고 계산
# skipna 옵션으로 조정 가능
df.mean(axis='columns', skipna=False)
# 축소 메서드의 옵션
# 옵션 설명
# axis 연산을 수행할 축, DataFrame 에서 0 은 로우고 1 은 컬럼
# skipna 누락된 값을 제외할 것인지 정하는 옵션, default 는 True
# level 계산하려는 측이 계층적 색인(다중 색인)이라면 레벨에 따라 묶어서 계산
# 최소값 또는 최대값을 가지는 색인 반환
df.idxmax()
df.idxmin()
# 누적합
df.cumsum()
# 한번에 여러 개의 통계 결과 반환
df.describe()
# 수치 데이터가 아닐 경우 describe 는 다른 요약 통계 생성
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()
# 요약 통계 관련 메서드
# 메서드 설명
# count NA 값을 제외한 값의 수
# describe 요약통계
# min, max 최소값, 최대값
# argmin, argmax 최소값, 최대값을 가지는 색인의 위치(정수)
# idxmin, idxmax 최소값, 최대값을 가지는 색인의 값
# quantile 분위수
# sum 합
# mean 평균
# median 중위값
# mad 평균절대편차
# prod 모든 값의 곱
# var 표본분산
# std 표본표준편차
# skew 왜도
# kurt 첨도
# cumsum 누적합
# cumin, cummax 누적최소값, 누적최대값
# cumprod 누적곱
# diff 차분
# pct_change 퍼센트 변화율

import pandas as pd

# 5.3.1 Correlation and Covariance
# pandas_datareader 패키지 이용
# conda install pandas-datareader
# pip install pandas-datareader
price = pd.read_pickle('C:/Users/You/Desktop/빅데이터/빅데이터 수업자료/파이썬 연습/새 폴더/yahoo_price.pkl')
volume = pd.read_pickle('C:/Users/You/Desktop/빅데이터/빅데이터 수업자료/파이썬 연습/새 폴더/yahoo_volume.pkl')
# 주가정보 다운로드 (may be not working)
import pandas_datareader.data as web
all_data = {ticker: web.get_data_yahoo(ticker) for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']}
price = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
volume = pd.DataFrame({ticker: data['Volume'] for ticker, data in all_data.items()})
# 각 주식의 퍼셋트 변화율 계산
returns = price.pct_change()
returns.tail()
# 상관관계 계산
returns['MSFT'].corr(returns['IBM'])
# 공분산 계산
returns['MSFT'].cov(returns['IBM'])
# 파이썬 속성 이름 규칙에 어긋나지 않아 좀 더 편리한 문법으로 해당 컬럼 선택 가능
returns.MSFT.corr(returns.IBM)
# DataFrame 에서 corr 과 cov 메서드는 DataFrame 행렬에서 상관관계와 공분산을 계산
returns.corr()
returns.cov()
# DataFrame 에서 corrwith 메서드 사용 시 다른 series 나 DataFrame 과의 상관관계를 계산
# Series 를 넘기면 각 컬럼에 대해 계산한 상관관계를 담고 있는 Series 반환
returns.corrwith(returns.IBM)
# DataFrame 을 넘기면 맞아떨어지는 컬럼 이름에 대한 상관관계 계산
returns.corrwith(volume)



# 5.3.2 Unique Values, Value Counts, and Membership
# 1 차원 Series 에 담긴 값의 정보를 추출하는 메서드
# 예시
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
# unique 메서드는 중복되는 값을 제거하고 유일값만 담고 있는 Series 를 반환
uniques = obj.unique()
uniques
# 유일값은 정렬된 순서대로 반환되지 않지만 필요하다면 uniques.sort()를 이용하여 정렬
obj.value_counts()
# 내림차순으로 정렬
# value_counts 메서드는 pandas 의 최상위 메서드로 어떤 배열이나 순차 자료구조에서도 사용 가능
pd.value_counts(obj.values, sort=False)
# isin 메서드는 어떤 값이 Series 에 존재하는지 나타내는 불리언 벡터 반환
# Series 나 DataFrame 의 컬럼에서 값을 골라내고 싶을 때 사용
obj
mask = obj.isin(['b', 'c'])
mask
obj[mask]
# Index.get_indexer 메서드는 여러 값이 들어 있는 배열에 유일한 값의 색인 배열을 구할 수 있다.
to_match = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])
unique_vals = pd.Series(['c', 'b', 'a'])
pd.Index(unique_vals).get_indexer(to_match)
# 유일값 세기, 멤버십 메서드
# 메서드 설명
# isin Series 의 각 원소가 넘겨받은 연속된 값에 속하는지 나타내는 불리언 배열 반환
# match 각 값에 대해 유일한 값을 담고 있는 배열에서의 정수 색인을 계산
# unique Series 에서 중복되는 값을 제거하고 유일값만 포함하는 배열 반환. 결과는 Series 에서 발견된 순서대로 반환
# value_counts Series 에서 유일값에 대한 색인과 도수 계산. 도수는 내림차순으로 정렬

# 히스토그램
data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4],
 'Qu2': [2, 3, 1, 2, 3],
 'Qu3': [1, 5, 2, 4, 4]})
data
result = data.apply(pd.value_counts).fillna(0)
result
# 여기서 결과값의 로우 라벨은 전체 컬럼의 유일한 값들을 담고 있다.
# 각 값은 각 컬럼에서 해당값이 몇 번 출현했는지 나타낸다.



