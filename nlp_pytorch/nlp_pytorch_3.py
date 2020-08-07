#딥러닝의 역사
#초기에는 규칙기반으로 시소러스같은 방식을 이용함
#이런 방식이 통계기반 기법으로 옮겨졌는데, CBOW, Bow, RNN, LSTM같은 기술을 뜻한다.
#한 층에 모든 통계에 기반한 결과를 가져오는 방식이다.

#2015년 seq2seq 방식이 소개되며 어텐션까지 도입되고 나서야 음성 인식분야에도 딥러닝으로 바뀌었다.


#단어 학습을 어렵게 하는 요소들 단어의 모호성(다의어), 다양한 표현, 불연속적인 데이터의 벡터화

import torch
import torch.nn as nn
import random
import numpy as np

x = torch.Tensor([[1,2],[3,4]])
#numpy.array() 와 비슷한 용도, 그래프와 경사도가 추가됨
#torch.Tensor == torch.FloatTensor
#x= torch.from_numpy(np.array([[1,2],[3,4]]))
#print(x)

#x= np.array([[1,2],[3,4]])
#print(x)

#autograd => 값을 앞으로 피드포워드하며 계산만해도, backward() 호출 한번에 역전파 알고리즘을 수행한다.
x = torch.Tensor(2,2)
y= torch.Tensor(2,2)
y.requires_grad_(True)
# 동적 그래프 자동으로 그래프가 생성된다. 최대 장점 그래프의 기울기 변화를 파악하기 좋다.
z = (x +y ) + torch.Tensor(2,2)


def liner(x,W,b):
    y = torch.mm(x,W) + b

    return y
x = torch.Tensor(16,10)
W = torch.Tensor(10,5)
b = torch.Tensor(5)
#print(x)
#print(W)
#print(b)
y = liner(x,W,b)
#print(y)

class MYLinear(nn.Module): ##nn.Module을 상속받은 클래스는 내부에 nn.module을 상속한 클래스 객체를 소유할 수 있다.
    def __init__(self, input_size, output_size):
        super().__init__()

        self.W = torch.Tensor(input_size,output_size)
        self.b = torch.Tensor(output_size)

    def forward(self, x):
        y = torch.mm(x,self.W)+ self.b #torch.mm은 피드 포워드함수이다.

        return y #이렇게 오버라이딩해서 피드포워딩하면 자동으로 오차역전파법을 해준다.

X = torch.Tensor(16,10)
#print("X: ",X)
linear = MYLinear(10,5)
y=linear.forward(X)
#print(y)
# nn.Parameters() 함수는 모듈 내에 선언된 학습이 필요한 파라미터들을 반환하는 이터레이터이다.
print("학습이 필요하다고 설정안한 경우: ",[p.size() for p in linear.parameters()]) #생각하기에는 W와 b 두 개가 학습이 필요할 것같은데 나오지 않는다.




#학습이 필요하다고 판단한 것은 따로 지정해 주어야 한다.
class MYLinear2(nn.Module): ##nn.Module을 상속받은 클래스는 내부에 nn.module을 상속한 클래스 객체를 소유할 수 있다.
    def __init__(self, input_size, output_size):
        super().__init__()

        self.W = nn.Parameter(torch.Tensor(input_size,output_size), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(output_size), requires_grad=True)
        #print("W: ",self.W)
        #print("b: ",self.b)

        #위의 식을 간단하게 구현하려면 nn.Linear()를 사용하면 된다.
        #nn.module을 상속받았기에 class내부에 nn 변수를 지정할 수 있다.
        #self.linear = nn.Linear(input_size,output_size)

    def forward(self, x):
        y = torch.mm(x,self.W)+ self.b #torch.mm은 피드 포워드함수이다.

        return y #이렇게 오버라이딩해서 피드포워딩하면 자동으로 오차역전파법을 해준다.

x2 = torch.Tensor(16,10)
linear2 =MYLinear2(10,5)
linear2.forward(x2)
print("학습 변수를 설정:",[x.size() for x in linear2.parameters()])


class MYLinear3(nn.Module): ##nn.Module을 상속받은 클래스는 내부에 nn.module을 상속한 클래스 객체를 소유할 수 있다.
    def __init__(self, input_size, output_size):
        super(MYLinear3,self).__init__()

        #nn.module을 상속받았기에 class내부에 nn 변수를 지정할 수 있다.
        self.linear = nn.Linear(input_size,output_size)

    def forward(self, x):
        y =self.linear(x)

        return y #이렇게 오버라이딩해서 피드포워딩하면 자동으로 오차역전파법을 해준다.
#이제 오차역전파법을 진행해보자.
target= 100
#정답 레이블 값
print("\n")
linear3 = MYLinear3(10,5)
y=linear3(x2) # == linear3.forward(x2) 왜 같은지 모르겠음
loss = (target - y.sum())**2
#print(loss)
#print(loss.backward()) #기울기 계산
#print(linear3.eval())
#print(linear3.train())

class MyModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(MyModel, self).__init__()

        self.linear = nn.Linear(input_size,output_size)

    def forward(self,X):
        y = self.linear(X)

        return y

def ground_truth(x):
    return 3*x[:,0]+x[:,1]-2*x[:,2]

def train(model, x, y,optim):
    optim.zero_grad() # 가중치 초기화

    y_hat = model(x) #feed-forward

    loss = ((y-y_hat)**2).sum()/x.size(0) # 손실함수 계산 직접 구현, 불러서 사용해도 된다.

    print(model.state_dict())

    params=model.state_dict()
    print(params['linear.weight'])
    print(params['linear.bias'])

    loss.backward()
    params = model.state_dict()
    print(params['linear.weight'])
    print(params['linear.bias'])

    optim.step() #오차역전파법 후에 최적화에 대해 step()까지 돌려야 가중치가 갱신된다.
    params=model.state_dict()
    print(params['linear.weight'])
    print(params['linear.bias'])

    #All optimizers implement a step() method, that updates the parameters. It can be used in two ways:
    return loss.data

batch_size = 1
n_epochs = 1000
n_iter = 10000

model = MyModel(3,1)

#결국은 미분이다 미분으로 최솟값을 찾는데 미분의 계산을 줄이기위해 back propagation을 사용합니다.
#back propagation의 효율을 높이기 위한 optimizer입니다. 학습률, momentum 등을 설정합니다.
optim = torch.optim.SGD(model.parameters(), lr= 1e-4, momentum=0.1) #손실함수에 대한 미분을 경사하강법으로 지정, 모멘톰 0.1 학습률 1e-4
#optimizer = optim.Adam([var1, var2], lr=0.0001) 아담 예시
#SGD = 확률적 경사하강법, 배치사이즈가 너무 커서 학습이 오래 걸릴때, 임의로 데이터셋을 뽑아내 학습시키는 것
x=torch.Tensor(1,3)
y=ground_truth(x.data)
loss = train(model, x, y, optim)


for epoch in range(n_epochs):
    avg_loss = 0

    for i in range(n_iter):
        x = torch.rand(batch_size, 3)
        y = ground_truth(x.data) # y는 레이블이기에 함수가 아니라 답을 넣어야 한다. x데이터를 집어넣어 값을 매긴다.

        loss = train(model, x, y,optim)

        avg_loss += loss #오차만큼 평균에 더한다.
        avg_loss = avg_loss / n_iter #첫번째

    x_valid = torch.Tensor([[.3,.2,.1]])
    y_valid = ground_truth(x_valid.data)

    #학습이 다 됐다고 판단하면 이제 평가를 위해 eval()함수 사용
    model.eval()
    y_hat = model(x_valid)
    model.train()

    print(avg_loss,y_valid.data[0],y_hat.data[0,0])

    if(avg_loss <0.001):
        break

#nn.Module 클래스를 상속받아 모델 아키텍처 클래스 선언
#해당 클래스 객체 생성
#SGD나 Adam 등의 옵티마이저를 생성하고, 생성한 모델의 파라미터를 최적화 대상으로 등록
#데이터로 미니배치를 구성하여 피드포워드 연산 그래프 생성
# 손실 함수를 통해 최종 결괏값과 손실값 계산
# 손실에 대해서 backward() 호출 -> 텐서들의 기울기가 채워짐
# 3번의 옵티마이저에서 step() 을 호출하여 경사하강법 1 스텝 수행
# 4번으로 돌아가 수렴 조건이 만족할때까지 반복