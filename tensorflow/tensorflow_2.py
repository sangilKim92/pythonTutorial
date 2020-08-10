import tensorflow as tf

I
''
W= tf.Variable(tf.randome_uniform([5,10],-1.0,1.0))
b= tf.Variable(tf.zeros([10]))

def dense(x):
    y=tf.matmul(W,x)+b
    return y

tf.keras.layers.Dense(...)
#Dense가 위의 신경망 기울기식을 만드는 식이다.
'''
Dense(units,activation=None,use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',
kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,**kwargs)
units: 출력 값의 크기, Integer, Long
activation: 호라성화 함수
use_bias : 편향 사용 여부
kernel_initializer: 가중치 초기화 함수
bias_initializer: 편향 초기화 함수
kernel_regularizer: 가중치 정규화 방법
bias_regularizer: 편향 정규화 방법
activity_regularizer: 출력값 정규화 방법
'''
# ex) 은닉층의 출력개수 10개, 출력층의 출력개수 2개 활성화함수는 각각 시그모이드, input은 20,1 크기의 실수형 값
NPUT_SIZE = (20,1)

input = tf.placeholder(tf.float32, shape=INPUT_SIZE)
'''
dropout은 여기서 실행
dropout = tf.keras.layers.Dropout(rate=0.2)(input)
hiidden(dropout)
'''
hidden = tf.keras.layers.Dense(units=10, activation=tf.nn.sigmoid)(input)
output = tf.keras.layers.Dense(units=2, activation=tf.nn.sigmoid)(hidden)


#Con1D, Con2D, Con3D
#기본적으로 이미지는 Con2D를 사용해서 특징을 뽑아내지만, 자연어는 (2,2)보단 1차원 배열에 적합하기에 Con1D를 사용한다.


INPUT_SIZE = (1,28,28)

input = tf.placeholder(tf.float32, shape=INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate=0.2)(input)

conv = tf.keras.layers.Conv1D(
    filters=10, # 필터의 개수
    kernel_size=3, #필터의 크기
    padding='same',
    activation=tf.nn.relu)(dropout)
max_pool = tf.keras.MaxPool1D(pool_size=3, padding='same')(conv)
flatten = tf.keras.layers.Flatten()(max_pool)
hidden = tf.keras.layers.Dense(units=50, activation=tf.nn.relu)(flatten)
output = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(hidden)