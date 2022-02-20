# 시계열 데이터는 금융, 경제, 신경과학 등 여러 다양한 분야에서 사용되는 매우 중요한 구조화된 데이터.
# 시간 상의 여러 지점을 관측하거나 측정할 수 있는 모든 것이 시계열
# 대부분의 시계열은 고정빈도(fixed frequency)로 표현되는데 데이터가 존재하는 지점이 15 초마다,
# 5 분마다 같은 특정 규칙에 따라 고정 간격을 가지게 된다.
# 어떻게 시계열 데이터를 표시하고 참조할지는 애플리케이션 의존적이며 다음의 한 유형일 수 있음
# 1) 시간 내에서 특정 순간의 타임스탬프
# 2) 2007 년 1 월이나 2020 년 전체 같은 고정된 기간
# 3) 시작과 끝 타임스탬프로 표시되는 시간 간격.
# 4) 실험 혹은 경과 시간, 각 타임스탬프는 특정 시작 시간에 상대적인 시간의 측정값

# 설정
import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)



### 11.1 Date and Time Data Types and Tools
# 파이썬 표준 라이브러리는 날짜와 시간을 위한 자료형과 달력 관련 기능을 제공하는 자료형이 존재
# datetime, datetime 형, 단순한 datetime 이 널리 사용
from datetime import datetime
now = datetime.now()
now
now.year, now.month, now.day
# datetime 은 날자와 시간을 모두 저장하며 마이크로초까지 지원
# datetime.timedelta 는 모두 datetime 객체 간의 시간적인 차이를 표현
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta
delta.days
delta.seconds
# timedelta 를 더하거나 빼면 그 만큼의 시간이 datetime 객체에 적용되어 새로운 객체를 만들 수 있다.
from datetime import timedelta
start = datetime(2011, 1, 7)
start + timedelta(12)
start - 2 * timedelta(12)

# datetime 모듈의 자료형
# 자료형      설명
# data       그레고리안 달력을 사용하여 날짜(연, 월, 일)를 저장
# time       하루의 시간을 시, 분, 초, 마이크로초 단위로 저장
# datetime   날짜와 시간을 저장
# timedelta  두 datetime 값 간의 차이(일, 초, 마이크로초)를 표현
# tzinfo     지역시간대를 저장하기 위한 기본 자료형


## 11.1.1 Converting Between String and Datetime
# datetime 객체와 pandas 의 Timestamp 객체는 str 메서드나 strftime 메서드에 포맷 규칙을 넘겨
# 문자열로 나타낼 수 있다.
stamp = datetime(2011, 1, 3)
str(stamp)
stamp.strftime('%Y-%m-%d')
# Datetime 포맷 규칙(ISO C89 호환)
# 포맷  설명
# %Y   4 자리 연도
# %y   2 자리 연도
# %m   2 자리 월[01, 12]
# %d   2 자리 일[01, 31]
# %H   시간(24 시간 형식) [00, 23]
# %I   시간(12 시간 형식) [01, 12]
# %M   2 자리 분[00, 59]
# %S   초[00, 61] (60, 61 은 윤초)
# %w   정수로 나타낸 요일[0(일요일),6]
# %U   연중 주차[00, 53]. 일요일을 그 주의 첫 번째 날로 간주. 그 해에서 첫 번째 일요일 앞에 있는 날은 0 주차
# %W   연중 주차[00, 53]. 월요일을 그 주의 첫 번째 날로 간주하며, 그 해에서 첫 번째 월요일 앞에 있는 날은 0 주차
# %z   UTC 시간대 오프셋을 +HHMM 또는 -HHMM 으로 표현. 시간대를 신경 쓰지 않는다면 비워둔다.
# %F   %Y-%m-%d 형식에 대한 축약(예: 2012-4-18)
# %D   %m/%d/%y 형식에 대한 축약(예: 04/18/12)
# 이 포맷코드는 datetime.strptime 을 사용해서 문자열을 날자로 변환할 때 사용할 수 있다.
value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')
datestrs = ['7/6/2011', '8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]
# datetime.strptime 은 알려진 형식의 날짜를 파싱하는 최적의 방법
# 하지만 매번 포맷 규칙을 써야 하는 건 귀찮은 일이다.
# 이 경우에는 서드파티 패키지인 dateutil 에 포함된 parser.parse 메서드를 사용하면 된다.(pandas 를 설치할 때 자동으로 함께 설치된다)

from dateutil.parser import parse
parse('2011-01-03')
# dateutil 은 거의 대부분의 사람이 인지하는 날짜 표현 방식을 파싱할 수 있다.
parse('Jan 31, 1997 10:45 PM')
# 국제 로케일의 경우 날짜가 월 앞에 오는 경우가 매우 흔하다.
# 이런 경우 dayfirst=True 를 넘겨주면 된다.
parse('6/12/2011', dayfirst=True)
# pandas 는 일반적으로 DataFrame 의 컬럼이나 축 색인으로 날짜가 담긴 배열을 사용한다.
# to_datetime 메서드는 많은 종류의 날짜 표현을 처리한다.
# ISO 8601 같은 표준 날짜 형식은 매우 빠르게 처리할 수 있다.
datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00']
pd.to_datetime(datestrs)
# 누락된 값(None, 빈 문자열 등)으로 간주되어야 할 값도 처리해 준다.
idx = pd.to_datetime(datestrs + [None])
idx
idx[2]
pd.isnull(idx)
# NaT(Not a Time)는 pandas 에서 누락된 타임스탬프 데이터를 나타낸다.
# datetime 객체는 여러나라 혹은 언어에서 사용하는 로케일에 맞는 다양한 포맷 옵션을 제공한다.

# 로케일별 날짜 포맷
# # 포맷  설명
# # %a   축약된 요일 이름
# # %A   요일 이름
# # %b   축약된 월 이름
# # %B   월 이름
# # %c   전체 날짜와 시간(예: 'Tue 01 May 2012 04:20:57 PM')
# # %p   해당 로케일에서 AM, PM 에 대응되는 이름(AM 은 오전, PM 은 오후)
# # %x   로케일에 맞는 날짜 형식(예: 미국이라면 2012 년 5 월 1 일은 '05/01/2012')
# # %X   로케일에 맞는 시간 형식(예: '04:24:12 PM')


### 11.2 Time Series Basics
# pandas 에서 찾아볼 수 있는 가장 기본적인 시계열 객체의 종류는 파이썬 문자열이나
# datetime 객체로 표현되는 타임스탬프로 색인된 Series
from datetime import datetime
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
         datetime(2011, 1, 7), datetime(2011, 1, 8),
         datetime(2011, 1, 10), datetime(2011, 1, 12)]

ts = pd.Series(np.random.randn(6), index=dates)
ts
# 내부적으로 보면 이들 datetime 객체는 DatetimeIndex 에 들어 있으며 ts 변수의 타입은 TimeSeries 다
ts.index
# 서로 다르게 색인된 시계열 객체 간의 산술 연산은 자동으로 날짜에 맞춰진다
ts + ts[::2]  # ts 에서 매 두번째 항목을 선택한다.
# pandas 는 Numpy 의 datetime64 자료형을 사용해서 나노초의 정밀도를 가지는 타임스탬프를 저장
ts.index.dtype
# DateTimeIndex 의 스칼라값은 pandas 의 Timestamp 객체
stamp = ts.index[0]
stamp
# Timestamp 는 datetime 객체를 사용하는 어떤 곳에도 대체 사용이 가능.
# 빈도에 대한 정보도 저장하며 시간대 변환을 하는 방법과 다른 종류의 조작을 하는 방법도 포함



## 11.2.1 Indexing, Selection, Subsetting
# 시계열은 데이터를 선택하고 인덱싱할 때 pandas.Series 와 동일하게 동작
stamp = ts.index[2]
ts[stamp]
# 해석할 수 있는 날짜를 문자열로 넘겨서 편리하게 사용 가능.
ts['1/10/2011']
ts['20110110']
# 긴 시계열에서는 연을 넘기거나 연, 월만 넘겨서 데이터의 일부 구간만 선택할 수 있다.
longer_ts = pd.Series(np.random.randn(1000),
 index=pd.date_range('1/1/2000', periods=1000))
longer_ts
longer_ts['2001'] # 연도로 해석되어 해당 기간의 데이터 선택
# 월에 대해서도 선택 가능
longer_ts['2001-05']
# datetime 객체로 데이터를 잘라내는 작업은 일반적인 Series 와 동일한 방식으로 할 수 있다.
ts[datetime(2011, 1, 7):]
# 범위를 지정하기 위해 시계열에 포함하지 않고 타임스탬프를 이용해서 Series 를 나눌 수 있다.
ts
ts['1/6/2011':'1/11/2011']
# 앞과 같이 날자 문자열이나 datetime 혹은 타임스탬프를 넘길 수 있다.
# 이런 방식으로 데이터를 나누면 NumPy 배열을 나누는 것처럼
# 원본 시계열에 대한 뷰를 생성한다는 사실을 기억하자
# 즉 데이터 복사가 발생하지 않고 슬라이스에 대한 변경이 원본 데이터에도 반영된다.
# 이와 동일한 인스턴스 메서드로 truncate 가 있는데, 이 메서드는 TimeSeries 를 두 개의 날짜로나눈다.
ts.truncate(after='1/9/2011')
# 위 방식은 DataFrame 에서도 동일하게 적용되어 로우에 인덱싱된다.
dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = pd.DataFrame(np.random.randn(100, 4),
                       index=dates,
                       columns=['Colorado', 'Texas', 'New York', 'Ohio'])
long_df.loc['5-2001']


## 11.2.2 Time Series with Duplicate Indices
# 여러 데이터가 특정 타임스탬프에 몰려 있는 경우
dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000', '1/3/2000'])
dup_ts = pd.Series(np.arange(5), index=dates)
dup_ts
# is_unique 속성을 통해 확인해 보면 색인이 유일하지 않음을 알 수 있다.
dup_ts.index.is_unique
# 이 시계열 데이터를 인덱싱하면 타임스탬프의 중복 여부에 따라 스칼라값이나 슬라이스가 생성된다.
dup_ts['1/3/2000'] # not duplicated
dup_ts['1/2/2000'] # duplicated
# 유일하지 않은 타임스탬프를 가지는 데이터를 집계하는 경우,
# 한 가지 방법은 groupby 에 level=0(단일 단계 인덱싱)을 넘기는 것
grouped = dup_ts.groupby(level=0)
grouped.mean()
grouped.count()


### 11.3 Date Ranges, Frequencies, and Shifting
# pandas 에서 일반적인 시계열은 불규칙적인 것으로 간주
# 시계열 안에서 누락된 값이 발생하더라도 일별, 월별 혹은 매 15 분 같은
# 상대적인 고정 빈도에서의 작업이 요구되는 경우가 종종 있다.
# pandas 에는 리샘플링, 표준 시계열 빈도 모음, 빈도 추론 그리고
# 고정된 빈도의 날짜 범위를 위한 도구가 있다.
# 예) 시계열을 고정된 일 빈도로 변환하려면 resample 메서드 사용
ts
resampler = ts.resample('D') # 'D'는 일 빈도로 해석
# 기본 빈도와 다중 빈도의 사용법을 살펴본다.


## 11.3.1 Generating Date Ranges
# pandas.date_range 를 사용하면 특정 빈도에 따라 지정한 길이만큼의 DatetimeIndex 생성
index = pd.date_range('2012-04-01', '2012-06-01')
index
# date_range 는 일별 타임스탬프를 생성
# 만약 시작 날짜나 종료 날짜만 넘긴다면 생성할 기간의 숫자를 함께 전달해야 한다.
pd.date_range(start='2012-04-01', periods=20)
pd.date_range(end='2012-06-01', periods=20)
# 시작과 종료 날짜는 생성된 날짜 색인에 대해 엄격한 경계를 정의
# 예) 날짜 색인 각 월의 마지막 영업일을 포함하도록 하고 싶다면
# 빈도값으로 'BM'(월 영업마감일)을 전달
pd.date_range('2000-01-01', '2000-12-01', freq='BM')


# 11-4    기본 시계열 빈도
# # 축약                          오프셋 종류             설명
# # D                                Day               달력상의 일
# # B                            BusinessDay           매 영업일
# # H                               Hour               매시
# # T 또는 min                      Minute              매분
# # S                              Second              매초
# # L 또는 ms                        Milli              밀리초(1/1000 초)
# # U                               Micro              마이크로초(1/1,000,000 초)
# # M                              MonthEnd            월 마지막 일
# # BM                         BusinessMonthEnd        월 영업마감일
# # MS                            MonthBegin           월 시작일
# # BMS                       BusinessMonthBegin       월 영업시작일
# # W-MON, W-TUE, ...               Week               요일, MON, TUE WED, THU, FRI, SAT, SUN
# # WOM-1MON, WOM-2MON, ...      WeekOfMonth           월별 주차와 요일. 예를 들어 WOM-3FRI는 매월 3 째주 금요일
# # Q-JAN, Q-FEB, ...            QuarterEnd            지정된 월을 해당년도의 마감으로 하며 지정된 월의 마지막 날짜를 가르키는 분기 주기
#                                                      (JAN, FEB, MAR, APR, MAY, JUN, JUL,AUG, SEP, OCT, NOV, DEC)
# # BQ-JAN, BQ-FEB, ...      BusinessQuarterEnd        지정된 월을 해당년도의 마감으로 하며 지정된월의 마지막 영업일을 가리키는 분기 주기
# # QS-JAN, QS-FEB, ...         QuarterBegin           지정된 월을 해당년도의 마감으로 하며 지정된 월의 첫째 날을 가리키는 분기 주기
# # BQS-JAN, BQS-FEB, ..    BusinessQuarterBegin       지정된 월을 해당년도의 마감으로 하며 지정된 월의 첫 번째 영업일을 가리키는 분기 주기
# # A-JAN, B-FEB, ...             YearEnd              주어진 월의 마지막 일을 가리키는 연간 주기
# #                                                    (JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC)
# # BA-JAN, BA-FEB, ...        BusinessYearEnd         주어진 월의 영업 마감일을 가리키는 연간 주기
# # AS-JAN, BS-FEB, ...          YearBegin             주어진 월의 시작일을 가리키는 연간 주기
# # BAS-JAN, BAS-FEB, ...     BusinessYearBegin        주어진 월의 영업 시작일을 가리키는 연간 주기

# date_range 는 기본적으로 시작 시간이나 종료 시간의 타임스탬프(존재한다면)를 보존
pd.date_range('2012-05-02 12:56:31', periods=5)
# 시간 정보를 포함하여 시작 날짜와 종료 날짜를 갖고 있으나
# 관계에 따라 자정에 맞추어 타임스탬프를 정규화하고 싶을 때
# normalize 옵션 사용
pd.date_range('2012-05-02 12:56:31', periods=5, normalize=True)



## 11.3.2 Frequencies and Date Offsets
# pandas 에서 빈도는 기본빈도(base frequency)와 배수의 조합으로 이루어진다.
# 기본 빈도는 보통 'M'(월별), 'H'(시간별)처럼 짧은 문자열로 참조된다.
# 각 기본 빈도에는 일반적으로 날짜 오프셋(date offset)이라고 불리는 객체를 사용할 수 있다.
# 예) 시간별 빈도는 Hour 클래스를 사용해서 표현할 수 있다.
from pandas.tseries.offsets import Hour, Minute
hour = Hour()
hour
# 이 오프셋의 곱은 정수를 넘겨서 구할 수 있다.
four_hours = Hour(4)
four_hours
# 대부분의 애플리케이션에서는 'H' 또는 '4H'처럼 문자열로 표현
# 기본 빈도 앞에 정수를 두면 해당 빈도의 곱을 생성
pd.date_range('2000-01-01', '2000-01-03 23:59', freq='4h')
# 여러 오프셋을 덧셈으로 합칠 수 있다.
Hour(2) + Minute(30)
# 빈도 문자열로 '1h30min'을 넘겨도 같은 표현으로 해석
pd.date_range('2000-01-01', periods=10, freq='2h30min')
# 어떤 빈도는 시간상에서 균일하게 자리 잡고 있지 않은 경우도 있다.
# 예) 'M'(월 마지막 일)은 월중 일수에 의존적이며
# 'BM'(월 영업마감일)은 월말이 주말인지 아닌지에 따라 다른다.
# 이를 책 저자는 앵커드(anchored) 오프셋이라고 부른다.
# Week of month dates (월별 주차)
# 유용한 빈도 클래스는 WOM 으로 시작하는 '월별 주차'
# 월별 주차를 사용하면 매월 3 번째주 금요일 같은 날짜를 얻을 수 있다.
rng = pd.date_range('2012-01-01', '2012-09-01', freq='WOM-3FRI')
list(rng)


## 11.3.3 Shifting (Leading and Lagging) Data
# 시프트는 데이터를 시간 축에서 앞이나 뒤로 이동하는 것을 의미
# Series 와 DataFrame 은 색인은 변경하지 않고
# 데이터를 앞으로 뒤로 느슨한 시프트를 수행하는 shift 메서드를 가지고 있다.
ts = pd.Series(np.random.randn(4), index=pd.date_range('1/1/2000', periods=4, freq='M'))
ts
ts.shift(2)
ts.shift(-2)
ts / ts.shift(1) - 1
# 이렇게 시프트를 하게 되면 시계열의 시작이나 끝에 결측치가 발생
# shift 는 일반적으로 한 시계열 내에서, 혹은 DataFrame 의 컬럼으로 표현할 수 있는 여러 시계열에서의 퍼센트 변화를 계산할 때 흔히 사용
# 코드로는 다음과 같이 표현
# ts / ts.shft(1)-1
# 느슨한 시프트는 색인을 바꾸지 않기 때문에 어떤 데이터는 버려지기도 한다.
# 그래서 만약 빈도를 알고 있다면 shft 에 빈도를 넘겨서 타임스탬프가 확장되도록 할 수 있다.
ts.shift(2, freq='M')
# 다른 빈도를 넘겨도 되는데, 이를 통해 아주 유연하게 데이터를 밀거나 당기는 작업을 할 수 있다.
ts.shift(3, freq='D')
ts.shift(1, freq='90T') # 'T'는 분을 나타냄
# Shifting dates with offsets (오프셋만큼 날짜 시프트)
# pandas 의 날짜 오프셋은 datetime 이나 Timestamp 객체에서도 사용할 수 있다.
from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
now + 3 * Day()
# 만일 MonthEnd 같은 앵커드 오프셋을 추가한다면 빈도 규칙의 다음 날짜로 롤 포워드(roll
forward)된다.
now + MonthEnd()
now + MonthEnd(2)
# 앵커드 오프셋은 rollforward 와 rollback 메서드를 사용해서 명시적으로 각각 날짜를 앞으로 밀거나 뒤로 당길 수 있다.
offset = MonthEnd()
offset.rollforward(now)
offset.rollback(now)
# 이 메서드를 groupby 와 함께 사용하면 날짜 오프셋을 영리하게 사용할 수 있다.
ts = pd.Series(np.random.randn(20),index=pd.date_range('1/15/2000', periods=20, freq='4d'))
ts
ts.groupby(offset.rollforward).mean()
# 가장 쉽고 빠른 방법은 resample 을 사용하는 것이다.
ts.resample('M').mean()