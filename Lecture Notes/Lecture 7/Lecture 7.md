# Lecture 7

### Vanishing Gradients and Fancy RNNs

---

### Overview : Vanishing Gradient 문제와 RNN의 여러 변형들

|Index|Subtitle|
|--- | --- |
|7.1.| Vanishing Gradient Problem |
|7.2.| New types of RNN : LSTM and GRU |
|7.3.| Miscellaneous fixes for vanishing gradients|
|7.4.| RNN variants : Bidirectional RNNs and Multi-layer RNNs |

---

### 7.1. Vanishing Grdient Problem

> **Vanishing Gradient에 대한 수학적 증명**

- RNN의 구조 속에서 Gradient를 계산할 때 값이 0으로 수렴함
- backpropagation된 값을 바탕으로 parameter를 update해야 하는데 먼 곳까지 gradient가 도달하지 못하고  0으로 수렴하면 input 근처에서 update가 이루어지지 않음 <br/>
<img src = '../../Figures/Lecture 7/Fig 1.jpg' width = "600dp"> <br/>
<img src = '../../Figures/Lecture 7/Fig 2.jpg' width = "600dp"> <br/>
- 위에서 나온 threshold 1은 sigmoid function의 nolinearlity 때문이며, tanh의 경우 1/4이 threshold가 됨 : [Paper](https://arxiv.org/pdf/1211.5063.pdf)
- Vanishing Gradient 때문에 모델의 weight/parameter는 '국소적인 효과'에 의해서만 업데이트됨
- 가까운 gradient signal만 도달할 수 있음

> **Vanishing Gradient에 대한 다른 설명**

- Gradient는 ***past가 future에 영향을 미치는 척도*** 로 이해할 수 있음
- RNN에서 longer distance를 가로지르는 과정에서 gradient가 소실되는 것의 의미는 ***step t, step t+n 사이에는 의존관계가 없다*** 는 의미
- 다시 말해 t와 t+n 번째 단어에 사이의 dependecy를 capture하지 못하는 잘못된 파라미터를 학습했음을 의미함
- RNN-LM 에서도 gradient가 long distance에서는 작으므로, long-distance dependency를 학습하지 못해서 거리가 먼 문맥을 활용해 단어를 예측하는 것이 어려움
- 예시 : *Syntactic recency : The writer of the books is(O)* vs *Sequential recency : The writer of the books are(X)*
    - 하지만 Vanishing Gradient 때문에 RNN-LM은 두 번째가 맞다고 결과를 도출함
    - Syntactic recency보다 Sequential recency를 학습하는 경향이 있음

> **Exploding gradient**

- Gradient가 굉장히 커질 경우, SGD update step에서 변화가 굉장히 커지게 됨
- Bad update를 야기함 : bad parameter configuration with large loss
- Inf 혹은 NaN 결과를 도출할 수도 있음
- 그 해결책으로 **Gradient Clipping**을 활용할 수 있음 <br/>
    <img src = '../../Figures/Lecture 7/Fig 3.jpg' width = "400dp"> <br/>
- Loss function을 3차원적으로 그렸을 때 절벽(cliff)이 있는 경우에는 gradient가 explode할 수 있다.
- 때문에 Gradient Clipping을 적용하여 gradient의 방향벡터는 유지하되, 크기를 threshold로 낮추어 gradient descent를 수행한다. <br/>
    <img src = '../../Figures/Lecture 7/Fig 4.jpg' width = "400dp"> <br/>
- RNN의 문제점 : RNN이 여러 timestep에 걸쳐서 학습한 정보를 보존하기란 쉽지 않음
- Vanilla RNN에서 hidden state가 끊임없이 'rewrite'되기 때문
- **RNN의 상태를 timestamp별로 저장하는 캐시 메모리**가 있다면 어떨까?

---

### 7.2. New types of RNN : LSTM and GRU

> **LSTM**

<img src = '../../Figures/Lecture 7/Fig 5.jpg' width = "700dp"> <br/>
- LSTM 구조는 RNN보다 더 긴 timestep에 걸친 정보를 보존할 수 있도록 한다.
    - 극단적으로 forget gate = 1(remember all)이라면 모든 cell의 정보가 영구적으로 저장될 것이다.
- LSTM은 vanishing/exploding gradient가 발생하지 않음을 보장하는 구조는 아니지만, long-distance dependency 학습에 용이한 구조임에는 분명하다.
- RNN에서 LSTM으로 구조를 변화시킨 것과 Vanishing/Explodig Gradient가 관련이 있나?
    - RNN에서 hidden state는 bottleneck 역할을 함 : 모든 gradient가 해당 state를 통과하게 됨
    - 그 state의 gradient 값이 작으면 backprop 과정에서 다음 gradient 역시 값이 작아질 수 밖에 없음
    - LSTM은 forget gate를 통해서 hidden state 거치지 않고 cell state에 반영될 수 있는 shortcut connection이 존재
- LSTM은 대부분의 NLP task에서 2013년~2015년까지 SOTA result를 계속해서 냄
    - Handwriting Recognition, Speech Recognition, Machine Translation, Parsing, Image captioning...
    - LSTM이 가장 dominant한 approach임
- 2019년부터는 Transformer 기반 접근이 dominant

> **GRU**

<img src = "../../Figures/Lecture 7/Fig 6.jpg" width = "700dp"><br/>
- GRU는 LSTM의 forget gate와 input gate를 reset gate, r(t) 하나로 처리(f+i=1)
- LSTM의 cell state와 hidden state 중에서 hidden state만 사용함
- GRU는 LSTM보다 computing 속도가 빠르고 파라미터 수가 적어 효율적이라는 것이 장점
- LSTM이 가장 standard한 선택이지만(특히 데이터가 long-dependency를 가지고 있거나 training data가 많을 경우), 효율적인 모델로 GRU를 활용할 수도 있음

---

### 7.3. Miscellaneous fixes for vanishing gradients

> **Is vanishing/explodig gradient just a RNN-problem?**

- Vanishing/Exploding Gradient 문제는 모든 인공신경망 아키텍처에서 나타날 수 있다.
- Feed-forward 그리고 convolutional 아키텍처와 무관하게, deep할수록 쉽게 발생한다.
- Chain Rule 때문에 근본적으로 발생하고, nonlinearlity function을 중간에 삽입하기 때문에 backpropagate하는 과정에서 gradient가 굉장히 작아지게 된다.
- 이로 인해 Lower Layer에서의 학습이 느리게 이루어진다.
- 해결을 위해 direct connection(shortcut)을 추가함으로써 gradient가 bottleneck을 거치치 않고도 직접 흘러갈 수 있도록 구성한다.
- **ResNet** : skip-connection을 통해서(residual connection), deep network를 train하기 쉽도록 한다.
- **HighwayNet** : highway connection을 활용
    - LSTM의 gate처럼 ResNet의 Identity connection을 조건부로 열고 닫음. 
    - Dynamic Gate에 의해서 Identity connection으로 정보를 바로 넘길 것인지, Transformation Layer를 거치게 할 것인지 결정함
- 결론적으로 Vanishing/Exploding Gradient는 일반적으로 deep network에서 나타나는 문제이지만, RNN이 특히 동일한 weight matrix를 반복적으로 곱하기 때문에 특히 취약하고 불안정하다.

---

### 7.4. RNN variants : Bidirectional RNNs and Multi-layer RNNs 

> **Bidirectional RNNs**

- Sentiment Classification Task를 수행한다고 가정해 보자.
    - 일반적인 RNN 아키텍처를 활용하게 되면 문장에서 왼쪽의 단어 임베딩을 먼저 학습하게 되고, 따라서 RNN에서의 output은 'left' context에 대한 정보를 담게 된다.
    - *The movie was terribly exciting!* 이라는 문장에 대한 감성분석을 수행할 때, 왼쪽에서 오른쪽으로 context를 누적하면 terribly 때문에 'negative' 결론을 도출한다.
    - 오른쪽 context에 대한 정보를 학습한다면 'exciting'과 결합된 'terribly'는 긍정의 의미가 되므로 bidirectional RNN이 필요하다.
    - Bidirectional RNN(LSTM, GRU)는 left hidden state를 학습한 다음 right hidden state를 학습해 left, right hidden state를 concatenate한 결과를 최종 hidden state로 활용한다.
<img src = '../../Figures/Lecture 7/Fig 7.jpg' width = "350dp">
- Entire Input Sequence가 확보된 경우에만 활용할 수 있으므로, Language Modeling처럼 Left Context만 존재하는 경우에는 적용 불가하다.
- 하지만 Entire Input Sequence가 확보된 경우에는 bidirectionality가 강력한 도구가 된다. 
- 대표적으로 BERT(Bidirectional Encoder Representations from Transformers)는 bidirectionality를 바탕으로 pretrained contextual representation system을 구축한 예시이다.

> **Mutli-layer RNNs**

- Multi-layer RNN은 주로 RNN의 성능 향상을 위해서 활용되었으며, RNN의 timestep-축으로의 stacking 외에도 다른 축으로 stack하여 활용한다.
- 각 Block을 iteration하는 순서가 같은 RNN layer 우선, 혹은 같은 Input 우선인지에 따라서 달라질 수 있으나 Pytorch 기준 같은 RNN layer를 우선적으로 순회하고 다음 RNN layer를 순회한다.
- Bidirectional Multi-layer RNN의 경우에는 같은 RNN layer를 우선적으로 반드시 순회해야 하므로 순서가 정해져 있다.
- CNN, DNN처럼 많은 layer를 쌓지는 않는다.
- Neural Machine Translation(NMT) Task에서는 2~4개의 layer가 encoder RNN, 4개의 layer가 decoder RNN으로 적합하다고 알려져 있다. [(2017, Britz et al.)](https://arxiv.org/pdf/1703.03906.pdf)
- Transformer 기반의 네트워크(BERT)는 24층까지 쌓는데, skipping-like connection을 활용해 Gradient Vanishing Problem in deep layers를 해결한다.