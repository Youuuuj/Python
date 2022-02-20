#### ch8 Data Wrangling: Join, Combine
import numpy as np
import pandas as pd
pd.options.display.max_rows = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

# 8.1 Hierarchical Indexing
# 계층적 색인: pandas 의 중요한 기능으로 다중(둘 이상) 색인 단계를 지정할 수 있게 해준다.
# 높은 차원의 데이터를 낮은 차원의 형식으로 다룰 수 있게 해주는 기능
# 리스트의 리스트(또는 배열)를 색인으로 하는 Series 생성
data = pd.Series(np.random.randn(9),
 index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
 [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data
# 생성한 객체가 MultiIndex 를 색인으로 하는 Series 인데, 색인의 계층을 보여주고 있다.
# 바로 윗 단계의 색인을 이용하여 하위 계층을 직접 접근 가능
data.index
# 계층적으로 색인된 객체는 데이터의 부분집합을 부분적 색인으로 접근하는 것이 가능
data['b']
data['b':'c']
data.loc[['b', 'd']]
# 하위 계층의 객체를 선택하는 것도 가능
data.loc[:,3]

# 계층적인 색인은 데이터를 재형성하고 피벗테이블 생성 같은 그룹 기반의 작업을 할 때 사용
# 예) 위에서 만든 DataFrame 객체에 unstack 메서드 사용하여 데이터를 새롭게 배열 가능
data.unstack()
# unstack 의 반대 작업은 stack 메서드로 수행
data.unstack().stack()

# DataFrame 에서는 두 축 모두 계층적 색인을 가질 수 있다.
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
 index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
 columns=[['Ohio', 'Ohio', 'Colorado'],
 ['Green', 'Red', 'Green']])
frame
# 계층적 색인의 각 단계는 이름(문자열이나 어떤 파이썬 객체라도 가능)을 가질 수 있고,
# 만약 이름을 가지고 있다면 콘솔 출력 시 함께 나타난다.
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
frame
# * 결과에서 색인 이름인 'state'와 'color'를 로우 라벨과 혼동하지 말 것!
# 컬럼의 부분집합을 부분적인 색인으로 접근하는 것도 컬럼에 대한 부분적 색인과 비슷하게 사용 가능
frame['Ohio']
# MultiIndex 는 따로 생성한 다음에 재사용이 가능
MultiIndex = data.index
# 위에서 살펴본 DataFrame 의 컬럼 계층이름은 다음처럼 생성할 수 있다
MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']],
 names=['state', 'color'])


# 8.1.1 Reordering and Sorting Levels
# 계층적 색인에서 계층의 순서를 바꾸거나 지정된 계층에 따라 테이터를 정렬해야 하는 경우
# swaplevel 은 넘겨받은 두 개의 계층 번호나 이름이 뒤바뀐 새로운 객체를 반환(데이터는 불변)
frame.swaplevel('key1', 'key2')
# sort_index 메서드는 단일 계층에 속한 데이터 정렬
# swaplevel 을 이용하여 계층을 바꿀때
# sort_index 를 사용하여 결과가 사전적으로 정렬 가능
frame.sort_index(level=1) # key2 기준 정렬
frame.swaplevel(0, 1).sort_index(level=0)
# * 객체가 계층적 색인으로 상위 계층부터 사전식으로 정렬되어 있다면(sort_index(level=0)이나
# sort_index()의 결과처럼) 데이터를 선택하는 성능이 훨씬 좋아진다.


# 8.1.2 Summary Statistics by Level
# DataFrame 과 Series 의 많은 기술 통계 및 요약 통계는 level 옵션을 가지고 있는데
# 한 축에 대해 합을 구하고 싶은 단계를 지정할 수 있는 옵션
frame.sum(level='key2')
frame.sum(level='color', axis=1)
# 8.1.3 Indexing with a DataFrame's columns
# DataFrame 에서 로우를 선택하기 위한 색인으로 하나 이상의 컬럼을 사용하거나
# 로우의 색인을 DataFrame 의 컬럼으로 옮기고 싶은 경우
frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
 'c': ['one', 'one', 'one', 'two', 'two',
 'two', 'two'],
 'd': [0, 1, 2, 0, 1, 2, 3]})
frame
# DataFrame 의 set_index()함수는 하나 이상의 컬럼을 색인으로 하는 새로운 DataFrame 을 생성
frame2 = frame.set_index(['c', 'd'])
frame2
# 컬럼을 명시적으로 남겨두지 않으면 DataFrame 에서 삭제된다.
frame.set_index(['c', 'd'], drop=False)
# reset_index()함수는 set_index()와 반대되는 개념
# 계층적 색인 단계가 컬럼으로 이동
frame2.reset_index()


# 8.2 Combining and Merging Datasets
# pandas 객체에 저장된 데이터는 다음과 같은 방법으로 합칠 수 있다.
# 1. pandas.merge 는 하나 이상의 키를 기준으로 DataFrame 의 로우를 합친다.
# SQL 이나 다른 관계형 데이터베이스의 join 연산과 유사
# 2. pandas.concat 은 하나의 축을 따라 객체를 이어붙인다.
# 3. combile_first 인스턴스 메서드는 두 객체를 포개서 한 객체에서 누락된 데이터를
# 다른 객체에 있는 값으로 채울 수 있도록 함.


# 8.2.1 Database-Style DataFrame Joins
# merge 나 join 연산은 하나 이상의 키를 사용하여 데이터 집합의 로우를 합친다.
# 예제(다대일)
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
 'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'],
 'data2': range(3)})
df1
df2
# merge()함수 사용
pd.merge(df1, df2)
# merge()함수는 중복된 컬럼을 키로 사용('key'컬럼)
pd.merge(df1, df2, on='key')
# 중복된 컬럼이 없는 경우 따로 지정
df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
 'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'],
 'data2': range(3)})
df3
df4
pd.merge(df3, df4, left_on='lkey', right_on='rkey')
# merge()함수는 기본적으로 inner join 수행
# how 인자로 'left', 'right', 'outer' 설정 가능
pd.merge(df1, df2, how='outer')
pd.merge(df1, df2, how='left')
pd.merge(df1, df2, how='right')
# how 옵션에 따른 다양한 조인 연산
# 옵션 동작
# 'inner' 양쪽 테이블 모두에 존재하는 키 조합 사용
# 'left' 왼쪽 테이블에 존재하는 모든 키 조합을 사용
# 'right' 오른쪽 테이블에 존재하는 모든 키 조합을 사용
# 'outer' 양쪽 테이블에 존재하는 모든 키 조합을 사용
# 다대다 merge
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
 'data1': range(6)})
df2 = pd.DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
 'data2': range(5)})
df1
df2
pd.merge(df1, df2, on='key', how='left')
# 다대다 join 은 두 로우의 데카르트곱 반환
pd.merge(df1, df2, how='inner')
# 여러 개의 키를 병합하려면 컬럼 이름이 담긴 리스트 이용
left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
 'key2': ['one', 'two', 'one'],
 'lval': [1, 2, 3]})
right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
 'key2': ['one', 'one', 'one', 'two'],
 'rval': [4, 5, 6, 7]})
pd.merge(left, right, on=['key1', 'key2'], how='outer')
# 여러 개의 키가 들어 있는 튜플의 배열이 단일 조인키로 사용된다
# * 컬럼과 컬럼을 조인할 때 전달한 DataFrame 객체의 색인은 무시된다.
# 겹치는 컬럼 이름에 대한 처리
# 1. 수동으로 컬럼이름을 겹치게 한다.
# 2. merge()함수에 있는 suffixes 인자로 두 DataFrame 객체에서 겹치는 컬럼 이름 뒤에 붙일 문자열 지정
pd.merge(left, right, on='key1')
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))

# merge 함수 인자 목록
# 인자 설명
# left 병합하려는 DataFrame 중 왼쪽에 위치한 DataFrame
# right 병합하려는 DataFrame 중 오른쪽에 위치한 DataFrame
# how 조인 방법. 'inner', 'outer', 'left', 'right'. 기본값은 'inner'
# on 조인하려는 컬럼 이름. 두 DataFrame 객체 모두에 존재하는 이름
# left_on 조인키로 사용할 left DataFrame 의 컬럼
# right_on 조인키로 사용할 right DataFrame 의 컬럼
# left_index 조인키로 사용할 left DataFrame 의 색인 로우(다중 색인인 경우 키)
# right_index 조인키로 사용할 right DataFrame 의 색인 로우(다중 색인인 경우 키)
# sort 조인키에 따라 병합된 데이터를 사전순으로 정렬. 기본값은 True
# suffixes 컬럼 이름이 겹칠 경우 각 컬럼 이름 뒤에 붙일 문자열의 튜플. 기본값은 ('_x', '_y')
# copy False 일 경우, 예외적인 경우에 데이터가 결과로 복사되지 않도록 한다. 기본값은 항상 복사.
# indicator merge 라는 이름의 특별한 컬럼을 추가하여 각 로우의 소스가 어디인지 나타낸다.
# 'left_only', 'right_only', 'both' 값을 가짐.


# 8.2.2 Merging on Index
# 병합하려는 키가 DataFrame 의 색인일 경우
# 이런 경우 left_index = True 또는 rignt_index = True (또는 둘다)를 지정하여
# 해당 색인을 병합키로 사용
left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
left1
right1
pd.merge(left1, right1, left_on='key', right_index=True)
# 외부 조인을 실행하여 합집합 구하기
pd.merge(left1, right1, left_on='key', right_index=True, how='outer')
# 계층 색인된 데이터 병합
lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio',
 'Nevada', 'Nevada'],
 'key2': [2000, 2001, 2002, 2001, 2002],
 'data': np.arange(5.)})
righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
 index=[['Nevada', 'Nevada', 'Ohio', 'Ohio',
 'Ohio', 'Ohio'],
 [2001, 2000, 2000, 2000, 2001, 2002]],
 columns=['event1', 'event2'])
lefth
righth
# 리스트로 여러 개의 컬럼을 지정하여 병합 (중복되는 색인값을 다룰 때는 how='outer'옵션 사용
pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)
pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')
# 양쪽에 공통적으로 존재하는 여러 개의 색인을 병합
left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
 index=['a', 'c', 'e'],
 columns=['Ohio', 'Nevada'])
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
 index=['b', 'c', 'd', 'e'],
 columns=['Missouri', 'Alabama'])
left2
right2
pd.merge(left2, right2, how='outer', left_index=True, right_index=True)
# 색인으로 병합할때 DataFrame 의 join 메서드 사용
left2.join(right2, how='outer')
# DataFrame 의 join 메서드는 왼쪽 조인을 수행
# join 메서드를 호출한 DataFrame 의 컬럼 중 하나에 대해 조인을 수행
left1.join(right1, on='key')
# 색인 대 색인으로 두 DataFrame 을 병합하려면 DataFrame 의 리스트에 join 메서드 사용
another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
 index=['a', 'c', 'e', 'f'],
 columns=['New York', 'Oregon'])
another
left2.join([right2, another])
left2.join([right2, another], how='outer')



# 8.2.3 Concatenating Along an Axis
# concatenation: 데이터를 합치는 또 다른 방법. binding, stacking 이라고 부름
# Numpy 는 ndarray 를 이어붙이는 concatenate()함수 제공
arr = np.arange(12).reshape((3, 4))
arr
np.concatenate([arr, arr], axis=1) # column 기준으로 병합
# Series 나 DataFrame 같은 pandas 객체의 컨텍스트 내부에는 축마다 이름이 있어서 배열을 쉽게 이어붙일 수 있다.
# pandas 의 concat()함수의 이용
# 예제 (색인이 겹치지 않는 3 개의 Series 객체)
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
s1
s2
s3
# 객체를 리스트로 묶어서 concat()함수에 전달하면 값과 색인을 연결
pd.concat([s1, s2, s3])
# concat()함수는 axis=0 을 기본값으로 새로운 Series 객체를 생성
# 만약 axis=1 을 넘긴다면 결과는 DataFrame 이 된다(axis=1 은 컬럼을 의미)
pd.concat([s1, s2, s3], axis=1)
# 겹치는 축이 없기 때문에 외부 조인으로 정렬된 합집합을 얻었지만
# join='inner'옵션을 사용하여 교집합을 구할 수도 있다
s4 = pd.concat([s1, s3])
s4
pd.concat([s1, s4], axis=1)
pd.concat([s1, s4], axis=1, join='inner')
# 'f'와 'g'라벨은 join='inner'옵션으로 사라짐
# join_axes 인자로 병합하려는 축을 직접 지정
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])  # join_axes : deprecated
# Series 를 이어붙이기 전의 개별 Series 를 구분할 수 없는 문제 발생
# 이어붙인 축에 대해 계층적 색인을 생성하여 식별이 가능하도록 할 수 있다.
# 계층적 색인을 생성하려면 keys 인자 사용
result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
result
result.unstack()
# Series 를 axis=1 로 병합할 경우 keys 는 DataFrame 의 컬럼명이 된다.
pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])
# DataFrame 객체에 대해서도 지금까지와 같은 방식 적용 가능
df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],
 columns=['one', 'two'])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],
 columns=['three', 'four'])
df1
df2
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])
# 리스트 대신 객체의 dict 를 이용하면 dict 의 키가 keys 옵션으로 사용된다.
pd.concat({'level1': df1, 'level2': df2}, axis=1)
# 계층적 색인을 생성할 때 사용할 수 있는 추가적인 옵션은 표 8-3 참조
# 새로 생성된 계층의 이름은 names 인자로 지정 가능
pd.concat([df1, df2], axis=1, keys=['level1', 'level2'],
 names=['upper', 'lower'])
# DataFrame 의 로우색인이 분석에 필요한 데이터를 포함하고 있지 않은 경우
df1 = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])
df1
df2
# 이 경우 ignore_index=True 옵션 사용
pd.concat([df1, df2], ignore_index=True) # row bind
# concat 함수 인자
# 인자 설명
# objs 이어붙일 pandas 객체의 사전이나 리스트. 필수 인자
# axis 이어붙일 축 방향. 기본값 0
# join 조인방식.기본값은 'outer'
# join_axes 합집합/교집합을 수행하는 대신 다른 n-1 축으로 사용할 색인 지정 --> deprecated
# keys 이어붙일 객체나 이어붙인 축에 대한 계층 색인을 생성하는데 연관된 값
# levels 계층 색인 레벨로 사용할 색인을 지정. key 가 넘어온 경우 여러 개의 색인을 지정
# names keyssk levels 혹은 둘 다 있을 경우 생성된 계층 레벨을 위한 이름
# verify_integrity 이어붙인 객체에 중복되는 축이 있는지 검사하고 있다면 예외 발생. 기본값은 False 로 중복 허용
# ignore_index 이어붙인 축의 색인을 유지하지 않고 range(total_length)로 새로운 색인을 생성


# 8.2.4 Combining Data with Overlap
# 두 데이터셋의 색인이 일부 겹치거나 전체가 겹치는 경우 데이터 병합이나 이어붙이기가 불가능
# 벡터화된 if-else 구문을 표현하는 NumPy 의 where()함수
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
 index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64),
 index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
a
b
np.where(pd.isnull(a), b, a) # null 이면 b, 아니면 a
# Series 객체의 combine_first 메서드는 위와 동일한 연산을 제공하며 데이터 정렬 기능까지 제공
b[:-2].combine_first(a[2:])
# DataFrame 에서 combine_first 메서드는 컬럼에 대해 같은 동작을 한다.
# 호출하는 객체에서 누락된 데이터를 인자로 넘긴 객체에 있는 값으로 채워 넣을 수 있다.
df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan],
 'b': [np.nan, 2., np.nan, 6.],
 'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.],
 'b': [np.nan, 3., 4., 6., 8.]})
df1
df2
df1.combine_first(df2) # NaN 또는 empty 자리에 df2 데이터 채우기


# 8.3 Reshaping and Pivoting
# 표 형식의 데이터를 재배치하는 기본 연산

# 8.3.1 Reshaping with Hierarchical Indexing
# 계층적 색인은 DataFrame 의 데이터를 재배치하는 다음의 방식 제공
# 1. stack: 데이터의 컬럼을 로우로 피벗(회전)
# 2. unstack: 로우를 컬럼으로 피벗
# 예제(문자열이 담긴 배열을 로우와 컬럼의 색인으로 하는 작은 DataFrame)
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
 index=pd.Index(['Ohio', 'Colorado'], name='state'),
 columns=pd.Index(['one', 'two', 'three'],
 name='number'))
data
# stack 메서드를 사용하면 컬럼이 로우로 피벗되어서 Series 객체를 반환
result = data.stack()
result
# unstack 메서드를 사용하면 위 계층적 색인을 가진 Series 로부터 다시 DataFrame 을 얻을 수 있다.
result.unstack()
# 기본적으로 가장 안쪽에 있는 레벨부터 끄집어 낸다(stack 도 동일)
# 레벨 숫자나 이름을 전달해서 끄집어낼 단계 지정 가능
result.unstack(0)
result.unstack(1)
result.unstack('state')
# 해당 레벨에 있는 모든 값이 하위그룹에 속하지 않을 경우
# unstack 하게 되면 누락된 데이터가 생길 수 있다.
s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s1
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
s2
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2
data2.unstack()
# stack 메서드는 누락된 데이터를 자동으로 걸려내기 때문에 연산을 쉽게 원상 복구할 수 있다.
data2.unstack()
data2.unstack().stack()
data2.unstack().stack(dropna=False)
# DataFrame 을 unstack()할 때 unstack 레벨은 결과에서 가장 낮은 단계가 된다.
df = pd.DataFrame({'left': result, 'right': result + 5},
 columns=pd.Index(['left', 'right'], name='side'))
df
df.unstack('state')
# stack 을 호출할 때 쌓을 축의 이름을 지정할 수 있다.
df.unstack('state').stack('side')


# 8.3.2 Pivoting “Long” to “Wide” Format
# 시계열 데이터는 일반적으로 시간 순서대로 나열
data = pd.read_csv('새 폴더/pandas_dataset2/macrodata.csv')
data.head()
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter,
 name='date')
columns = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
data = data.reindex(columns=columns)
data.index = periods.to_timestamp('D', 'end')
ldata = data.stack().reset_index().rename(columns={0: 'value'})
# PeriodIndex 는 시간 간격을 나타내기 위한 자료형
# 연도(year)와 분기(quarter)컬럼을 합친다.
# ldata 는 긴 형식
# 여러 시계열이나 둘 이상의 키(예제에서는 date 와 item)를 가지고 있는 다른 관측 데이터에서 사용
# 각 로우는 단일 관측치를 나타낸다.
ldata[:10]
# 관계형 데이터베이스는 테이블에 데이터가 추가되거나 삭제되면 item 컬럼에 별개의 값을 넣거나 빼는 방식으로
# 고정된 스키마(컬럼 이름과 데이터형)에 데이터 저장
# 위의 예에서 date 와 item 은 기본키(primary key)가 되어 관계무결성을 제공하며
# 쉬운 조인 연산과 프로그램에 의한 질의를 가능하게 해준다.
# 길이가 긴 형식으로는 작업이 용이하지 않을 수 있어서
# 하나의 DataFrame 에 date 컬럼의 시간값으로 색인된 개별 item 을 컬럼으로 포함시키는 것을 선호할지도 모른다.
# DataFrame 의 pivot 메서드가 이런 변형을 지원
pivoted = ldata.pivot('date', 'item', 'value')
pivoted
# pivot 메서드의 첫 두 인자는 로우와 컬럼 색인으로 사용될 컬럼명
# 마지막 두 인자는 DataFrame 에 채워 넣을 값을 담고 있는 컬럼 이름
# 한 번에 두개의 컬럼을 동시에 변형
ldata['value2'] = np.random.randn(len(ldata))
ldata[:10]
# 마지막 인자를 생략해서 계층적 컬럼을 가지는 DataFrame 을 얻울 수 있다.
pivoted = ldata.pivot('date', 'item')
pivoted[:5]
pivoted['value'][:5]
# pivot 은 단지 set_index 를 사용해서 계층적 색인을 만들고
# unstack 메서드를 이용해서 형태를 변경하는 단축키 같은 메서드
unstacked = ldata.set_index(['date', 'item']).unstack('item')
unstacked[:7]



# 8.3.3 Pivoting “Wide” to “Long” Format
# pivot 과 반대되는 연산은 pandas.melt
# 하나의 컬럼을 여러 개의 새로운 DataFrame 으로 생성하기보다는 여러 컬럼을 하나로 병합하고
# DataFrame 을 입력보다 긴 형태로 만들어 낸다.
# 예제
df = pd.DataFrame({'key': ['foo', 'bar', 'baz'],
 'A': [1, 2, 3],
 'B': [4, 5, 6],
 'C': [7, 8, 9]})
df
# 'key'컬럼을 그룹 구분자로 사용할 수 있고 다른 컬럼을 데이터값으로 사용할 수 있다.
# pandas.melt 를 사용할 때는 반드시 어떤 컬럼을 그룹 구분자로 사용할 것인지 지정해야 한다.
# 여기서는 'key'를 그룹 구분자로 지정
melted = pd.melt(df, ['key'])
melted
# pivot 을 사용하여 원래 모양으로 되돌릴 수 있다.
reshaped = melted.pivot('key', 'variable', 'value')
reshaped
# pivot 의 결과로 로우 라벨로 사용하던 컬럼에서 색인을 생성하므로
# reset_index 를 이용해서 데이터를 다시 컬럼으로 되돌려놓는다.
reshaped.reset_index()
# 데이터값으로 사용할 컬럼들의 집합을 지정할 수도 있다.
pd.melt(df, id_vars=['key'], value_vars=['A', 'B','C'])
# pandas.melt 는 그룹 구분자 없이도 사용할 수 있다.
pd.melt(df, value_vars=['A', 'B', 'C'])
pd.melt(df, value_vars=['key', 'A', 'B'])