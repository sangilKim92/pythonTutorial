import re
'''
정규식
메타데이터 :원래의 뜻말고 특별한 의미로 사용되는 문자들
. ^ $ * + ? { } [ ] \ | ( ) 
[abc] => []안의 a,b,c중 한 개의 문자와 매치하라는 뜻
[a-zA-Z0-9]
[]안에 어떤 것도 사용될 수 있지만, ^만 다른 의미로 사용됨. []안의 식은 하나하나씩 찾아줌
[^0-9] =>0~9를 제외한 모든 문자 매치
A.a = A2a, Aba, 등 A와 a사이에 모든 걸 허용 (except \n)
A.*a = A와 a사이에 문자가 길어도 상관없음
\d - 숫자와 매치, [0-9]와 동일한 표현식이다.
\D - 숫자가 아닌 것과 매치, [^0-9]와 동일한 표현식이다.
\s - whitespace 문자와 매치, [ \t\n\r\f\v]와 동일한 표현식이다. 맨 앞의 빈 칸은 공백문자(space)를 의미한다.
\S - whitespace 문자가 아닌 것과 매치, [^ \t\n\r\f\v]와 동일한 표현식이다.
\w - 문자+숫자(alphanumeric)와 매치, [a-zA-Z0-9_]와 동일한 표현식이다.
\W - 문자+숫자(alphanumeric)가 아닌 문자와 매치, [^a-zA-Z0-9_]와 동일한 표현식이다.

+ 1번이상의 반복
* 0번이상의 반복
{n:m} n번부터 m번까지 반복
? 0, 1번 존재

re 모듈 method
pattern 또는 re.compile 을 이용해서 정규식을 컴파일한다.

match() 문자열의 처음부터 정규식과 매치되는지 조사  - 처음이 다르다면 None 맞다면 match 객체
search() : 문자열 전체를 검색해서 정규식과 매치되는지 조사 -맞다면 match객체 아니면 none
findall() 정규식과 매치되는 모든 문자열를 리스트로 반환
finditer() 정규식과 매치되는 모든 문자열을 반복 가능한 객체로 반환

match 메소드의 객체 메소드
group(): 매치된 문자열을 반환한다.
start() 매치된 문자열의 시작 위치를 반환한다.
end() 매치된 문자열의 끝 위치를 알려준다.
span() 매치된 문자열의 시작과 끝을 튜플로 알려준다.

re.I 대소문자 구결을 없앤다.


'''

p = re.compile('[1-9a-z]*\s*[1-9a-z]*') #*를 작성해서 여러개 뽑아내고 \s로 띄어쓰기 포함 뒤도 마찬가지
m = p.match( 'string goes here' )
if m:
    print('Match found: ', m.group())
else:
    print('No match')

re.compile('[a-z]')
result = p.findall("life is too short")
print(result)

p = re.compile('[a-z]', re.I) #대소문자 무시
p.match('python')
p.match('Python')
p.match('PYTHON')


p = re.compile("^python\s\w+", re.MULTILINE) #^는 문자열의 처음, $은 문자열의 끝을 의미한다.
                                            #원래는 문자열의 처음인 python one만 출력하지만
                                            #re.MULTILINE을 옵션으로 넣었기에 각 라인이 python으로 
                                            #시작하는 것들을 가져온다.

data = """python one
life is too short
python two
you need python
python three"""

print(p.findall(data))