# 유준

# 12
# 1. 수행과제
# pandas와 numpy를 import하고, pandas의 Series와 DataFrame을 네임스페이스로 import 하시오.
# random seed는 1234로 설정하시오.

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
np.random.seed(1234)


# 2. 수행과제 : 다음의 데이터프레임을 구성하고 "debt" 컬럼이 Null인지 확인하시오.
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame2 = pd.DataFrame(data,
                      columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four','five', 'six'])

pd.isnull(frame2['debt'])
frame2['debt'].isnull()

# 3. 수행과제 : 다음의 문자열 리스트를 Series로 변환하고 diplay하시오.
# ['aardvark', 'artichoke', , 'avocado'] (3번째 자리 빈칸)
pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])


# 4~6 : 다음의 DataFrame을 구성하고 문제에서 필요한 python coding을 하시오.
data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=['Ohio', 'Colorado', 'Utah', 'NewYork'],
    columns=['one', 'two', 'three', 'four'])

# 4. 수행과제 : 'three'컬럼의 값이 5초과인 데이터를 display하시오.
data[data['three'] > 5]

# 5. 수행과제 : 'four' 컬럼을 삭제하고 결과를 display하시오.
del data['four']
data

# 6. 수행과제 : 'Utah' 로우를 삭제하고 결과를 display하시오.
data.drop(['Utah'])


# 7~8 : 다음의 DataFrame을 구성하고 문제에서 필요한 python coding을 하시오.
frame = pd.DataFrame(
    np.arange(8).reshape((2, 4)),
    index=['three', 'one'],
    columns=['d', 'a', 'b', 'c'])

# 7. 수행과제 : 로우 기준으로 오름차순으로 정렬하시오.
frame.sort_index(axis=0, ascending=True)   # ascending=True가 default. 오름차순
# 8. 수행과제 : 컬럼기준으로 내림차순으로 정렬하시오.
frame.sort_index(axis=1, ascending=False)


# 9~10. 아래 리스트를 이용하여 문제에서 필요한 python coding을 하시오.
# [1., , 3.5, , 7] (2번째 및 4번째 자리 빈칸)
# 9. 수행과제 : Series로 변환한 후 display 하시오.
data = pd.Series([1.,np.nan, 3.5, np.nan, 7])
# 10. 수행과제 : NA(빈칸)을 제외한 숫자들의 평균을 빈칸에 추가하고 display 하시오.
data.fillna(data.mean())


# 11~13. 다음의 데이터프레임을 구성하고 문제에서 필요한 python coding을 하시오.
np.random.seed(1234)
data = pd.DataFrame(np.random.randn(1000, 4))

# 11. 수행과제 : 왼쪽에서 3번째 컬럼을 display하시오
data[2]
# 12. 수행과제 : 데이터의 절대값이 3이상인 데이터는 데이터의 부호(+/-)의 3배 한 값으로 대체하시오.
data[np.abs(data) > 3] = np.sign(data) * 3
# 13. 수행과제 : 해당 객체의 데이터의 각종 통계량을 요약해서 출력해 주는 메서드를 사용하여 요약하여 보이시오.
data.describe()

# 14~15. 다음의 데이터프레임을 구성하고 문제에서 필요한 python coding을 하시오.
df = pd.DataFrame({'key': ['foo', 'bar', 'baz'],
                   'A': [1, 2, 3],
                   'B': [4, 5, 6],
                   'C': [7, 8, 9]})
# 14. 수행과제 : 'key'를 그룹 구분자로 지정하여 wide form을 long form으로 변환하고 display하시오.
melted = pd.melt(df, ['key'])
melted
# 15. 수행과제 : pivot을 사용하여 long form에서 wide form으로 변환(원래의 데이터프레임으로 변환)하고 display하시오.
reshaped = melted.pivot('key', 'variable', 'value')
reshaped