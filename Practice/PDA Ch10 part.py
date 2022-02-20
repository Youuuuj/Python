# ch10. Data Aggregation and Group Operations
import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# 10.1 GroupBy Mechanics
# by Hadley Wickham
# split-apply-combine 이라는 그룹연산에 대한 용어
# 그룹연산 단계
# 1. Series, DataFrame 같은 pandas 객체나 다른 객체에 들어 있는 데이터를 하나 이상의 키를 기준으로 분리
# 2. 객체는 하나의 축을 기준으로 분리하고 나서 함수를 각 그룹에 적용시켜 새로운 값을 얻는다.
# 3. 함수를 적용한 결과를 하나의 객체로 결합한다.
# 각 그룹의 색인은 다음의 다양한 형태가 될 수 있으며, 모두 같은 타입일 필요도 없다.
# - 그룹으로 묶을 축과 동일한 길이의 리스트나 배열
# - DataFrame 의 컬럼 이름을 지칭하는 값
# - 그룹으로 묶을 값과 그룹 이름에 대응하는 사전이나 Series 객체
# - 축 색인 혹은 색인 내의 개별 이름에 대해 실행되는 함수
# DataFrame 으로 표현되는 간단한 표 형식의 데이터
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
 'key2' : ['one', 'two', 'one', 'two', 'one'],
 'data1' : np.random.randn(5),
 'data2' : np.random.randn(5)})
df
# 이 데이터를 key1 으로 묶고 각 그룹에서 data1 의 평균을 구하기
grouped = df['data1'].groupby(df['key1'])
grouped

# 이 grouped 변수는 GroupBy 객체
# 이 객체는 그룹 연산을 위해 필요한 모든 정보를 가지고 있어서 각 그룹에 어떤 연산을 적용할 수 있게 해준다.
# 그룹별 평균을 구하기 위해 GroupBy 객체의 mean 메서드 사용
grouped.mean()
# 새롭게 생성된 Series 객체의 색인은 'key1'인데, 그 이유는 DataFrame 컬럼인 df['key1'] 때문
# 만약 여러 개의 배열을 리스트로 넘겼다면 조금 다른 결과를 얻었을 것
means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means
# 데이터를 두개의 색인으로 묶었고, 그 결과 계층적 색인을 가지는 Series 를 얻음.
means.unstack()
# 이 예제에서 그룹의 색인 모두 Series 객체인데, 길이만 같다면 어떤 배열도 상관없다.
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()
# 한 그룹으로 묶을 정보는 주로 같은 DataFrame 안에서 찾게 된다.
# 이 경우 컬럼명(문자열, 숫자 혹은 다른 파이썬 객체)을 넘겨서 그룹의 색인으로 사용할 수 있다.
df.groupby('key1').mean()
df.groupby(['key1', 'key2']).mean()
# 위에서 df.groupby('key1').mean()코드를 보면 key2 컬럼이 결과에서 빠져있다.
# 그 이유는 df['key2']는 숫자 데이터가 아니기 때문에 이런 컬럼은 nuisance column 이라고 부르며 결과에서 제외
# 기본적으로 모든 숫자 컬럼이 수집되지만 원하는 부분만 따로 걸러내는 것도 가능
# GroupBy 메서드는 그룹의 크기를 담고 있는 Series 를 반환하는 size 메서드
df.groupby(['key1', 'key2']).size()
# *그룹 색인에서 누락된 값은 결과에서 제외된다.


# 10.1.1 Iterating Over Groups
# GroupBy 객체는 iteration 을 지원. 그룹이름과 그에 따른 데이터 묶음을 튜플로 반환
for name, group in df.groupby('key1'):
 print(name)
 print(group)
# 색인이 여럿 존재하는 경우 튜플의 첫 번째 원소가 색인값이 된다.
for (k1, k2), group in df.groupby(['key1', 'key2']):
 print((k1, k2))
 print(group)
# 이 안에서 원하는 데이터만 골라낼 수 있다.
# 한 줄이면 그룹별 데이터를 사전형(dict)으로 변환하여 사용 가능
pieces = dict(list(df.groupby('key1')))
pieces['b']
# groupby 메서드는 기본적으로 axis=0 에 대해 그룹을 만든다
# 다른 축으로 그룹을 만드는 것도 가능
# 예제에서 df 의 컬럼을 dtype 에 따라 그룹으로 묶을 수 있다.
df.dtypes
grouped = df.groupby(df.dtypes, axis=1)
# 그룹을 아래처럼 출력 가능
for dtype, group in grouped:
 print(dtype)
 print(group)


# 10.1.2 Selecting a Column or Subset of Columns
# DataFrame 에서 만든 GroupBy 객체를 컬럼 이름이나 컬럼 이름이 담긴 배열로 색인하면 수집을 위해 해당
# 컬럼을 선택하게 된다.
df.groupby('key1')['data1']
df.groupby('key1')[['data2']]
# 아래 코드도 같은 결과 산출
df['data2'].groupby(df['key1'])
df[['data2']].groupby(df['key1'])
# 대용량 데이터를 다룰 경우 소수의 컬럼만 집계하고 싶은 경우
df.groupby(['key1', 'key2'])[['data2']].mean()
# 색인으로 얻은 객체는 groupby 메서드에 리스트나 배열을 넘겼을 경우
# DataFrameGroupBy 객체가 되고, 단일 값으로 하나의 컬럼 이름만 넘겼을 경우 SeriesGroupBy 객체가 된다.
s_grouped = df.groupby(['key1', 'key2'])['data2']
s_grouped
s_grouped.mean()


# 10.1.3 Grouping with Dicts and Series
# 그룹 정보는 배열이 아닌 형태로 존재하기도 한다.
# DataFrame 예제
people = pd.DataFrame(np.random.randn(5, 5),
 columns=['a', 'b', 'c', 'd', 'e'],
 index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.iloc[2:3, [1, 2]] = np.nan # Add a few NA values
people
# 각 컬럼을 나타낼 그룹 목록이 있고 그룹별로 컬럼의 값을 모두 더하는 경우
mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
 'd': 'blue', 'e': 'red', 'f': 'orange'}
# dict 에서 groupby 메서드로 배열을 추출할 수 있지만
# 이 dict 에 groupby 메서드를 적용
by_column = people.groupby(mapping, axis=1)
by_column.sum()
# Series 에 대해서도 같은 기능 수행 가능
map_series = pd.Series(mapping)
map_series
people.groupby(map_series, axis=1).count()


# 10.1.4 Grouping with Functions
# 그룹 색인으로 넘긴 함수는 색인값 하나마다 한 번씩 호출되며,
# 반환값은 그 그룹의 이름으로 사용
# 이전 예제에서 people DataFrame 은 사람의 이름을 색인값으로 사용.
# 만약 이름의 길이별로 그룹을 묶고 싶다면 일므의 길이가 담긴 배열을 만들어 넘기는 대신

# len()함수 사용
people.groupby(len).sum()
# 내부적으로 모두 배열로 변환되므로 함수를 배열, dict 또는 Series 와 섞어 쓰더라도 문제가 되지않는다.
key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()


# 10.1.5 Grouping by Index Levels
# 계층적으로 색인된 데이터는 축 색인의 단계 중 하나를 사용해서 편리하게 집계할 수 있는 기능 제공
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
 [1, 3, 5, 1, 3]],
 names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
hier_df
# 이 기능을 사용하려면 level 예약어를 사용해서 레벨 번호나 이름을 사용
hier_df.groupby(level='cty', axis=1).count()



## 10.2 Data Aggregation
# 데이터 집계: 배열로부터 스칼라값을 만들어내는 모든 데이터 변환 작업
# 표 10-1 에 있는 것과 같이 일반적인 데이터 집계는 데이터 묶음에 대한 준비된 통계를 계산해내는 최적화된 구현을 가지고 있다.
# 표 10-1 최적화된 group by 메서드
# 함수 설명
# count 그룹에서 NA 가 아닌 값의 수를 반환
# sum NA 가 아닌 값들의 합을 구한다
# mean NA 가 아닌 값들의 평균을 구한다
# median NA 가 아닌 갓ㅂ들의 산술 중간값을 구한다
# std, var 편향되지 않은(n-1 을 분모로 하는) 표준편차와 분산
# min, max NA 가 아닌 갓ㅂ들 중 최솟값과 최댓값
# prod NA 가 아닌 값들의 곱
# first, last NA 가 아닌 값들 중 첫째 값과 마지막 값
# 직접 고안한 집계함수를 사용하고 추가적으로 그룹 객체에 이미 정의된 메서드를 연결해서 사용하는것도 가능
# 예로, quntile 메서드가 Series 나 DataFrame 의 컬럼의 변위치를 계산한다는 점을 가정하자
# quantile 메서드는 GroupBy 만을 위해 구현된건 아니지만 Series 메서드 이기 때문에사용가능하다.
# 내부적으로 GroupBy 는 Series 를 효과적으로 잘게 자르고 각 조각에 대해 piece.quantile(0.9)를 호출한다.
# 그리고 이 결과들을 모두 하나의 객체로 합쳐서 반환한다.
df
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)  # 백분위수 보여줌  90% 값 보여줌
# 자신의 데이터 집계함수를 사용하려면 배열의 aggregate 나 agg 메서드에 해당 함수를 넘기면 된다.
def peak_to_peak(arr):
 return arr.max() - arr.min()
grouped.agg(peak_to_peak)
# describe 메서드는 데이터를 집계하지 않는데도 잘 작동함을 확인할 수 있다.
grouped.describe()
# *사용자 정의 집계함수는 일반적으로 표 10-1 에 있는 함수에 비해 느리게 동작하는데
# 그 이유는 중간 데이터를 생성하는 과정에서 함수 호출이나 데이터 정렬 같은 오버헤드가 발생하기때문


# 10.2.1 Column-Wise and Multiple Function Application
# 앞에서 살펴본 팁 데이터를 다시 고려하자
# 여기서 read_csv()함수로 데이터를 불러온 다음 팁의 비율을 담기 위한 컬럼인 tip_pct 를 추가
tips = pd.read_csv('파이썬 연습/새 폴더/pandas_dataset2/tips.csv')
# Add tip percentage of total bill
tips['tip_pct'] = round(tips['tip'] / tips['total_bill']*100,2)
tips[:6]
# Series 나 DataFrame 의 모든 컬럼을 집계하는 것은 mean 이나 std 같은 메서드를 호출하거나 원하는 함수에
# aggregate 를 사용하는 것이다.
# 하지만 컬럼에 따라 다른 함수를 사용해서 집계를 수행하거나 여러 개의 함수를 한 번에 적용하기 원한다면
# 이를 쉽게 수행할 수 있다.
# tips 를 day 와 smoke 별로 묶어보자.
grouped = tips.groupby(['day', 'smoker'])
# 표 10-1 에서의 함수 이름을 문자열로 넘기면 된다.
grouped_pct = grouped['tip_pct']
grouped_pct.agg('mean')
# 만약 함수 목록이나 함수 이름을 넘기면 함수 이름을 컬럼 이름으로 하는 DataFrame 을 얻는다
grouped_pct.agg(['mean', 'std', peak_to_peak])
# 여기서는 데이터 그룹에 대해 독립적으로 적용하기 위해 agg 에 집계함수들의 리스트를 넘겼다.
# GroupBy 객체에서 자동으로 지정하는 컬럼 이름을 그대로 쓰지 않아도 된다.
# lamda 함수는 이름이 '<lamda>'인데 이를 그대로 쓸 경우 알아보기 힘들어진다.
# 이때 이름과 함수가 담긴(name, function) 튜플의 리스트를 넘기면
# 각 튜플에서 첫 번째 원소가 DataFrame 에서 컬럼 이름으로 사용된다.
# (2 개의 튜플을 가지는 리스트가 순서대로 매핑된다.)
grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])
# DataFrame 은 컬럼마다 다른 함수를 적용하거나 여러 개의 함수를 모든 컬럼에 적용할 수 있다.
# tip_pct 와 total_bill 컬럼에 대해 동일한 세 가지 통계를 계산한다고 가정
functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
result
# 위에서 반환된 DataFrame 은 계층적인 컬럼을 가지고 있으며 이는 각 컬럼을 따로 계산한 다음
# concat 메서드를 이용해서 keys 인자로 컬럼 이름을 넘겨서 이어 붙인 것과 동일하다.
result['tip_pct']
# 위에서 처럼 컬럼 이름과 메서드가 담긴 튜플의 리스트를 넘기는 것도 가능
ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
grouped['tip_pct', 'total_bill'].agg(ftuples)
# 컬럼마다 다른 함수를 적용하고 싶다면 agg 메서드에 커럼 이름에 대응하는 함수가 들어있는 dict 를 넘기면 된다.
grouped.agg({'tip' : np.max, 'size' : 'sum'})
grouped.agg({'tip_pct' : ['min', 'max', 'mean', 'std'], 'size' : 'sum'})

# 10.2.2 Returning Aggregated Data Without Row Indexes
# 지금까지 예제에서 집계된 데이터는 유일한 그룹키 조합으로 색인(어떤 경우에는 계층적 색인)되어 반환되었다.
# groupby 메서드에 as_index=False 를 넘겨서 색인되지 않도록 할 수 있다.
tips.groupby(['day', 'smoker'], as_index=False).mean()
# 색인된 결과에 대해 reset_index 메서드를 호출해서 같은 결과를 얻을 수 있다.
# as_index=False 옵션을 사용하면 불필요한 계산을 피할 수 있다.


## 10.3 Apply: General split-apply-combine
# apply 메서드는 객체를 여러조각으로 나누고, 전달된 함수를 각 조각에 일괄 적용한 후 이를 다시합친다.
# 팁 데이터에서 그룹별 상위 5 개의 tip_pct 값을 골라보자.
# 우선 특정 컬럼에서 가장 큰 값을 가지는 로우를 선택하는 함수를 작성
def top(df, n=0, column='tip_pct'):
 return df.sort_values(by=column)[-n:]
top(tips, n=6)
# 흡연자(smoker) 그룹에 대해 이 함수(top)를 apply 하면 다음과 같은 결과 산출
tips.groupby('smoker').apply(top)
# 위 결과를 보면 top 함수가 나뉘어진 DataFrame 의 각 부분에 모두 적용 되었고,
# pandas.concat 을 이용하여 하나로 합쳐진 다음 그룹 이름표가 붙었다.
# 결과는 계층적 색인을 가지게 되고 내부 색인은 원본 DataFrame 의 색인값을 가지게 된다.
# 만일 apply 메서드로 넘길 함수가 추가적인 인자를 받는다면 함수 이름 뒤에 붙여서 넘겨주면 된다.
tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')
# 앞에서 GroupBy 객체에 describe 메서드를 호출한 적이 있다.
result = tips.groupby('smoker')['tip_pct'].describe()
result
result.unstack('smoker')
# describe 같은 메서드를 호출하면 GroupBy 내부적으로 다음과 같은 단계를 수행
f = lambda x: x.describe()
grouped.apply(f)


# 10.3.1 Suppressing the Group Keys
# 앞의 예제에서 반환된 객체는 원본 객체의 각 조각에 대한 색인과 그룹 키가 계층적 색인으로 사용됨을 볼 수 있다.
# 이런 결과는 groupby 메서드에 group_keys=False 로 설정하여 막을 수 있다.
tips.groupby('smoker', group_keys=False).apply(top)

# 10.3.2 Quantile and Bucket Analysis
# ch8 에서 pandas 의 cut 과 qcut 메서드를 사용해서 선택한 크기만큼 또는 표본 변위치에 따라 데이터를 나눌 수 있었다.
# 이 함수들을 groupby 와 조합하면 데이터 묶음에 대해 변위치 분석이나 버킷 분석을 매우 쉽게 수행할 수 있다.
# 임의의 데이터 묶음을 cut 을 이용해서 등간격 구간으로 나누어 보자.
frame = pd.DataFrame({'data1': np.random.randn(1000),
 'data2': np.random.randn(1000)})
quartiles = pd.cut(frame.data1, 4)
quartiles[:10]
def get_stats(group):
 return {'min': group.min(), 'max': group.max(),
 'count': group.count(), 'mean': group.mean()}
grouped = frame.data2.groupby(quartiles)
grouped.apply(get_stats).unstack()
# Return quantile numbers
grouping = pd.qcut(frame.data1, 10, labels=False)
grouped = frame.data2.groupby(grouping)
grouped.apply(get_stats).unstack()


# 10.3.3 Example: Filling Missing Values with Group-Specific Values
s = pd.Series(np.random.randn(6))
s[::2] = np.nan
s
s.fillna(s.mean())
states = ['Ohio', 'New York', 'Vermont', 'Florida',
 'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4
data = pd.Series(np.random.randn(8), index=states)
data
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
data
data.mean()
data.groupby(group_key).mean()
fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)
fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda g: g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)


# 10.3.4 Example: Random Sampling and Permutation
# Hearts, Spades, Clubs, Diamonds
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
 cards.extend(str(num) + suit for num in base_names)

deck = pd.Series(card_val, index=cards)
deck[:13]

def draw(deck, n=5):
 return deck.sample(n)
draw(deck)
get_suit = lambda card: card[-1] # last letter is suit
deck.groupby(get_suit).apply(draw, n=2)
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)


# 10.3.5 Example: Group Weighted Average and Correlation
df = pd.DataFrame({'category': ['a', 'a', 'a', 'a',
 'b', 'b', 'b', 'b'],
 'data': np.random.randn(8),
 'weights': np.random.rand(8)})
df
grouped = df.groupby('category')
np.average(df['data'], weights=df['weights'])
get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
grouped.apply(get_wavg)
close_px = pd.read_csv('파이썬 연습/새 폴더/pandas_dataset2/stock_px_2.csv', parse_dates=True, index_col=0)
close_px.info()
close_px[-4:]
spx_corr = lambda x: x.corrwith(x['SPX'])
rets = close_px.pct_change().dropna()
get_year = lambda x: x.year
by_year = rets.groupby(get_year)
by_year.apply(spx_corr)
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))


# 10.3.6 Example: Group-Wise Linear Regression
# install 'statsmodels' 라이브러리
import statsmodels.api as sm
def regress(data, yvar, xvars):
 Y = data[yvar]
 X = data[xvars]
 X['intercept'] = 1.
 result = sm.OLS(Y, X).fit()
 return result.params
by_year.apply(regress, 'AAPL', ['SPX'])