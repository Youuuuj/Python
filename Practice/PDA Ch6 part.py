# PDA ch6 part
# chapter6 Data Loading, Storage
import numpy as np
import pandas as pd

# 환경설정
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

# 6.1 Reading and Writing Data in Text Format
# read_csv() 와 read_table() 사용
# 표 6-1 pandas 파일 파싱 함수
# 함수 설명
# read_csv 파일, url 또는 파일과 유사한 객체로부터 구분된 데이터를 읽어온다. 구분자는 쉼표(,)를 기본으로 한다.
# read_table 파일, url 또는 파일과 유사한 객체로부터 구분된 데이터를 읽어온다. 구분자는 탭('\t')를 기본으로 한다.
# read_fwf 고정폭 컬럼 형식에서 데이터를 읽어온다(구분자가 없는 데이터)
# read_clipboard 클립보드에 있는 데이터를 읽어오는 read_table(), 웹페이지에서 표를 읽어올때 유용
# read_excel 엑셀파일(xls, xlsx)에서 표 형식의 데이터를 읽어온다.
# read_hdf pandas 에서 저장한 HDFS 파일에서 데이터를 읽어온다.
# read_html HTML 문서 내의 모든 테이블의 데이터를 읽어온다.
# read_json JSON 문자열에서 데이터를 읽어온다.
# read_msgpack 메세지팩 바이너리 포맷으로 인코딩된 pandas 데이터를 읽어온다.
# read_pickle 파이썬 피클 포맷으로 저장된 객체를 읽어온다.
# read_sas SAS 시스템의 사용자 정의 저장 포맷으로 저장된 데이터를 읽어온다.
# read_sql SQL 쿼리 결과를 pandas 의 DataFrame 형식으로 읽어온다.
# read_stata Stata 파일에서 데이터를 읽어온다.
# read_feather Feather 바이너리 파일 포맷으로부터 데이터를 읽어온다.
# 위 함수들은 텍스트 데이터를 DataFrame 으로 읽어오기 위한 함수
# 아래의 옵션을 가짐.

# 6.1 Reading and Writing Data in Text Format
# read_csv() 와 read_table() 사용
# 표 6-1 pandas 파일 파싱 함수
# 함수 설명
# read_csv 파일, url 또는 파일과 유사한 객체로부터 구분된 데이터를 읽어온다. 구분자는 쉼표(,)를 기본으로 한다.
# read_table 파일, url 또는 파일과 유사한 객체로부터 구분된 데이터를 읽어온다. 구분자는 탭('\t')를 기본으로 한다.
# read_fwf 고정폭 컬럼 형식에서 데이터를 읽어온다(구분자가 없는 데이터)
# read_clipboard 클립보드에 있는 데이터를 읽어오는 read_table(), 웹페이지에서 표를 읽어올때 유용
# read_excel 엑셀파일(xls, xlsx)에서 표 형식의 데이터를 읽어온다.
# read_hdf pandas 에서 저장한 HDFS 파일에서 데이터를 읽어온다.
# read_html HTML 문서 내의 모든 테이블의 데이터를 읽어온다.
# read_json JSON 문자열에서 데이터를 읽어온다.
# read_msgpack 메세지팩 바이너리 포맷으로 인코딩된 pandas 데이터를 읽어온다.
# read_pickle 파이썬 피클 포맷으로 저장된 객체를 읽어온다.
# read_sas SAS 시스템의 사용자 정의 저장 포맷으로 저장된 데이터를 읽어온다.
# read_sql SQL 쿼리 결과를 pandas 의 DataFrame 형식으로 읽어온다.
# read_stata Stata 파일에서 데이터를 읽어온다.
# read_feather Feather 바이너리 파일 포맷으로부터 데이터를 읽어온다.

# 위 함수들은 텍스트 데이터를 DataFrame 으로 읽어오기 위한 함수
# 아래의 옵션을 가짐.
# 색인: 반환하는 DataFrame에서 하나 이상의 컬럼을 색인으로 지정할 수 있다.
# 파일이나 사용자로부터 컬럼 이름을 받거나 아무것도 받지 않을 수 있다.
# 자료형 추론과 데이터 변환: 사용자 정의 값 변환과 비어 있는 값을 위한 사용자 리스트 포함
# 날짜분석: 여러 컬럼에 걸쳐 있는 날짜와 시간 정보를 하나의 컬럼에 조합해서 결과에 반영
# 반복: 여러 개의 파일에 걸쳐 있는 자료를 반복적으로 읽어올 수 있다.
# 정제되지 않은 데이터 처리: 로우나 꼬리말, 주석 건너뛰기 또는 천 단위마다 쉼표로 구분된 숫자 같은 것 처리
# pandas.read_csv()는 데이터 형식에 자료형이 포함되어 있지 않아 type추론을 수행 # HDF5나 Feather, msgpack의 경우 데이터 형식에 자료형 포함


# type examples/ex1.csv
df = pd.read_csv('새 폴더/ex1.csv')
df

# 구분자를 쉼표로 지정
pd.read_csv('새 폴더/ex1.csv', sep=',')

# type examples/ex2.csv
pd.read_csv('새 폴더/pandas_dataset2/ex2.csv', header=None)

# 컬럼명 지정
pd.read_csv('새 폴더/pandas_dataset2/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])

# message컬럼을 색인으로 하는 DAtaFrame을 반환하려면 index_col인자에 4번째 컬럼
# 또는 'message'이름을 가진 컬럼을 지정하여 색인으로 만듦
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('새 폴더/pandas_dataset2/ex2.csv', names=names, index_col='message')

# type examples/csv_mindex.csv
# 계층적 색인 지정 시 컬럼 번호나 이름의 리스트를 넘긴다.
parsed = pd.read_csv('새 폴더/pandas_dataset2/csv_mindex.csv', index_col=['key1', 'key2'])
parsed

# 구분자 없이 공백이나 다른 패턴으로 필드를 구분
list(open('새 폴더/pandas_dataset2/ex3.txt'))

# 공백문자로 구분되어 있는 경우 정규표현식 \s+사용
result = pd.read_table('새 폴더/pandas_dataset2/ex3.txt', sep='\s+')
result

# 첫번째 로우는 다른 로우보다 컬럼이 하나 적기 때문에 read_table 은 첫번째 컬럼은 DataFrame 의 색인으로 인식
# skiprows 를 이용하여 첫번째, 세번째, 네번째 로우를 건너뛴다.
pd.read_csv('새 폴더/pandas_dataset2/ex4.csv', skiprows=[0, 2, 3])

# 텍스트파일에서 누락된 값은 표기되지 않거나(비어 있는 문자열) 구분하기 쉬운 특수한 문자로 표기
# 기본적으로 pandas 는 NA 나 NULL 처럼 흔히 통용되는 문자를 비어있는 값으로 사용
result = pd.read_csv('새 폴더/pandas_dataset2/ex5.csv')
result
pd.isnull(result)

# na_values 옵션은 리스트나 문자열 집합을 받아서 누락된 값 처리
result = pd.read_csv('새 폴더/pandas_dataset2/ex5.csv', na_values=['NULL'])
result

# 컬럼마다 다른 NA 문자를 사전값으로 넘겨서 처리
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('새 폴더/pandas_dataset2/ex5.csv', na_values=sentinels)

# 표 6-2 read_csv 와 read_table 함수 인자
# 인자 설명
# path 파일시스템에서의 위치. URL,파일객체를 나타내는 문자열
# sep 또는 delimiter 필드를 구분하기 위해 사용할 연속된 문자나 정규 표현식
# header 컬럼이름으로 사용할 로우 번호. 기본값은 0(첫번째 로우). 헤더가 없을 경우에는 None 으로 지정가능
# index_col 색인으로 사용할 컬럼 번호나 이름. 계층적 색인을 지정할 경우 리스트를 넘길 수 있다.
# names 컬럼이름으로 사용할 리스트. header=None 과 함께 사용
# skiprows 파일의 시작부터 무시할 행 수 또는 무시할 로우 번호가 담긴 리스트
# na_values NA 값으로 처리할 값들의 목록
# comment 주석으로 분류되어 파싱하지 않을 문자 혹은 문자열
# parse_dates 날짜를 datetime 으로 변환할지 여부. 기본값은 False. True 인 경우
# 모든 컬럼에 적용된다.
# 컬럼의 번환 이름을 포함한 리스트를 넘겨서 변환할 컬럼을 지정 가능
# keep_date_col 여러 컬럼을 datetime 으로 변환했을 경우 원래 컬럼을 남겨둘지 여부. default 는 True
# converters 변환 시 컬럼에 적용할 함수 지정 예) {'foo': f}는 'foo'컬럼에 f 함수 적용
# dayfirst 모호한 날짜 형식일 경우 국제 형식으로 간주. 기본값은 False
# date_parser 날짜 변환 시 사용할 함수
# nrows 파일의 첫 일부만 읽어올 때 처음 몇 줄을 읽을 것인지 지정
# iterator 파일을 조금씩 읽을 때 사용하도록 TextParser 객체를 반환하도록 함. 기본값은 False.
# chunksize TextParser 객체에서 사용할 한 번에 읽을 파일의 크기
# skip_footer 파일의 끝에서 무시할 라인 수
# verbose 파싱 결과에 대한 정보 출력. 숫자가 아닌 값이 들어 있는 컬럼에 누락된 값이 있다면 줄 번호를 출력. default 는 False
# encoding 유니코드 인코딩 종류 지정. 'utf-8'
# sqeeze 컬럼이 하나뿐인 경우 Series 객체를 반환. default 는 False
# thousands 숫자를 천 단위로 끊을 때 사용

# 6.1.1 Reading Text Files in Pieces
# 큰 파일을 다루기 전에 pandas 의 출력 설정
pd.options.display.max_rows = 10
# 최대 10 개의 데이터 출력
result = pd.read_csv('새 폴더/pandas_dataset2/ex6.csv')
result
# 처음 몇줄만 읽을 때 nrows 옵션 사용
pd.read_csv('새 폴더/pandas_dataset2/ex6.csv', nrows=5)
# 파일을 여러 조각으로 나누어서 읽고 싶다면 chunksize 옵션으로 로우 개수 설정
chunker = pd.read_csv('새 폴더/pandas_dataset2/ex6.csv', chunksize=1000)
print(chunker)
# read_csv 에서 반환된 TetParser 객체를 이용해서 chunksize 에 따라 분리된 파일들을 순회할 수있다
# 예로 ex6.csv 파일을 순회하면서 'key'로우에 있는 값을 세어보려면 다음과 같이 한다.
chunker = pd.read_csv('새 폴더/pandas_dataset2/ex6.csv', chunksize=1000)
tot = pd.Series([])
for piece in chunker:
 tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False)
tot[:10]


# 6.1.2 Writing Data to Text Format
# 데이터를 구분자로 구분한 형식으로 내보내기
data = pd.read_csv('새 폴더/pandas_dataset2/ex5.csv')
data
data.to_csv('out.csv')
# 다른 구분자 사용도 가능
import sys
data.to_csv(sys.stdout, sep='|')
# 결과에서 누락된 값은 비어 있는 문자열로 나타나는데 원하는 값으로 지정 가능
data.to_csv(sys.stdout, na_rep='NULL')
# 다른 옵션을 명시하지 않으면 로우와 컬럼 이름이 기록된다. 로우와 컬럼 이름을 포함하지 않을 경우 아래와 같이 사용
data.to_csv(sys.stdout, index=False, header=False)
# 컬럼의 일부분만 기록하거나 순서 지정
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])
# Series 에도 to_csv 메서드 존재
dates = pd.date_range('1/1/2000', periods=7)
ts = pd.Series(np.arange(7), index=dates)
ts.to_csv('새 폴더/pandas_dataset2/tseries.csv')
# type examples/tseries.csv


# 6.1.3 Working with Delimited Formats
# type examples/ex7.csv
# pandas_read_table() 함수를 이용하여 대부분의 파일 형식을 불러 올 수 있다.
# csv 파일을 불러오는 경우
import csv
f = open('새 폴더/pandas_dataset2/ex7.csv')
reader = csv.reader(f)
print(list(reader))
# 큰 따옴표가 제거된 튜플 얻을 수 있다.
for line in reader:
 print(line)
# 원하는 형태로 데이터를 넣을 수 있도록 하자.
# 파일을 읽어 줄 단위 리스트로 저장
with open('새 폴더/pandas_dataset2/ex7.csv') as f:
 lines = list(csv.reader(f))
print(lines)

# 헤더와 데이터 구분
header, values = lines[0], lines[1:]
# 사전표기법과 로우를 컬럼으로 전치해주는 zip(*values)이용 데이터 컬럼 사전 만들기
data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict
# csv 파일은 다양한 형태로 존재할 수 있다. 다양한 구분자, 문자열을 둘러싸는 방법, 개행 문자 같은 것들은
# csv.Dialect 를 상속받아 새로운 클래스를 정의해서 해결
class my_dialect(csv.Dialect):
 lineterminator = '\n'
 delimiter = ';'
 quotechar = '"'
 quoting = csv.QUOTE_MINIMAL
# reader = csv.reader(f, dialect=my_dialect)
reader = csv.reader('새 폴더/pandas_dataset2/ex7.csv', dialect=my_dialect)
print(list(reader))
# 서브클래스를 정의하지 않고 csv.readr 에 키워드 인자로 각각의 csv 파일의 특징을 지정해서 전달해도 된다.
# reader = csv.reader(f, delimiter='|')
reader = csv.reader('새 폴더/pandas_dataset2/ex7.csv', delimiter='|')
# 사용가능한 옵션(csv.Dialect 의 속성)

# CSV 관련 옵션
# 인자 설명
# delimiter 필드를 구분하기 위한 한 문자로 된 구분자. 기본값은 ','
# lineterminator 파일을 저장할 때 사용할 개행문자. 기본값은 '\r\n'. 파일을 읽을 때는 이 값을 무시.
# quotechar 각 필드에서 값을 둘러싸고 있는 문자. 기본값은 '"'.
# quoting 값을 읽거나 쓸때 둘러쌀 문자 컨벤션. csv.QUOTE_ALL(모든 필드에 적용),
# csv.QUOTE_MINIMAL(구분자 같은 특별한 문자가 포함된 필드만 적용), csv.QUOTE_NONE(값을 둘러싸지 않음)옵션이 있다.
# skipinitialspace 구분자 뒤에 있는 공백 문자를 무시할 지 여부. 기본값은 False.
# doublequote 값을 둘러싸는 문자가 필드 내에 존재할 경우 처리 여부. True 일 경우 그 문자까지 모두 둘러싼다.
# escapechar quoting 이 csv.QUOTE_NONE 일때 값에 구분자와 같은 문자가 있을 경우 구별할 수 있도록 해주는 이스케이프 문자
# ('\'같은). 기본값은 None
# CSV 처럼 구분자로 구분된 파일을 기록하려면 csv.writer 를 이용하면 된다.
# csv.writer 는 이미 열린, 쓰기가 가능한 파일 개체를 받아서 csv.reader 와 동일한 옵션으로 파일을 기록
with open('새 폴더/pandas_dataset2/mydata.csv', 'w') as f:
 writer = csv.writer(f, dialect=my_dialect)
 writer.writerow(('one', 'two', 'three'))
 writer.writerow(('1', '2', '3'))
 writer.writerow(('4', '5', '6'))
 writer.writerow(('7', '8', '9'))
