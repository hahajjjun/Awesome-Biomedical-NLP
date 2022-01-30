# Lecture 8

### Translation, Seq2Seq, Attention

---

### Overview : Machine Translation을 위한 모델들

|Index|Subtitle|
|--- | --- |
|8.1.| Pre-Neural Machine Translation |
|8.2.| Neural Machine Translation : Seq2Seq |
|8.3.| Neural Technique : Attention |

---

### 8.1. Pre-Neural Machine Translation

> **Machine Translation**

- Machine Translation(MT)는 source language로 쓰여진 문장 x를 target language로 쓰여진 문장 y로 변환하는 task이다.
- 초기의 Machine Translation은 대부분 규칙 기반으로, bilingual dictionary를 만들어 단어 하나하나를 대응시키는 방식이다.
- 그래서 *Statistical Machine Translation(SMT)*라고 부른다.
- Data로부터 확률적인 모델을 학습하게 되고, Naive Bayes와 비슷한 접근에 해당한다.
   <img src = "../../Figures/Lecture 8/Eq 1.svg" width = "300dp"></br>
- 여기서 앞의 항은 Translation Model이 도출하고, 뒤의 항은 Language Model이 도출한다.
- Language Model은 ***monolingual data*** 로 학습하고, ***fluency*** 즉 target language를 활용해 학습해서 문장의 구성을 유창하게 배치하도록 학습한다.
- Translational Model은 ***bilingual parallel data*** 로 학습하고, ***fidelity*** 즉 부분적인 문장의 구(phrase), 단어(word) 수준에서 어떻게 번역할지를 학습한다.

> **Translation Model을 통한 P(x|y)의 학습**

- P(x|y)를 학습하기 위해서 Translation Model은 P(x , a|y)를 학습한다. 
- 특정 target 단어 y가 주어졌을 때의 alignment, 그리고 source 단어 x가 대응될 확률을 계산한다.
- alignment는 source와 target 언어 사이에서의 correspondence를 의미하며, 굉장히 복잡하다.
    <img src = "../../Figures/Lecture 8/Fig 1.jpg" width = "500dp"></br>
- P(x, a|y)를 학습한다는 것의 의미는 y(target)이 주어졌을 때 특정한 단어 alignment가 등장할 확률을 계산하고, 특정한 target 단어 y와 대응되는 source 단어 x의 개수가 몇 개일지에 대한 확률을 계산하는 과정 등으로 구성된다. 
    <img src = "../../Figures/Lecture 8/Eq 1.svg" width = "300dp"></br>
- 이 항을 최대화하는 y를 어떻게 찾을까? 모든 target vocabulary를 순회하면서 최댓값을 찾는 것도 방법이지만, expensive한 방법이다.
- **Heuristic Search Algorithm**을 통해서 확률이 너무 낮은 y는 제거하는 방식으로 비용을 절약할 수 있다.
- 이렇게 주어진 probability를 최대화하는 y를 찾는 과정을 *decoding*이라 하며, 이후 Beam Search 등의 알고리즘을 언급할 예정이다.
- Statistcial Machine Translation(SMT) 모델들 중 좋은 성능을 내는 모델은
    -복잡한 구성을 가진다.
    - 여러 subcomponent로 구성되어 있다.
    - Feature Engineering 과정이 많이 필요하다.
    - Extra Resources & Human Effort가 많이 든다.

---

### 8.2. Neural Machine Translation : Seq2Seq

> **Neural Machine Translation**

- Neural Machine Translation(NMT)는 Single Neural Network를 활용해 Machine Translation을 수행하는 방법론이다.
- NMT를 위한 Neural Network 아키텍처는 **sequence-to-sequence**, **seq2seq**이라고 부르며 두 개의 RNN(LSTM, GRU)로 구성되어 있다.
- 인코더 RNN와 디코더 RNN으로 구성된 seq2seq 아키텍처<br/>
<img src = "../../Figures/Lecture 8/Fig 2.jpg" width = "500dp"> <br/>
- seq2seq는 Machine Translation 외에도 Summarization, Dialogue, Parsing, Code Generation같은 Generative한 Task에도 활용된다.
    - Summary = long text >> short text
    - Dialogue = Previous utterance >> Next utterance
    - Parsing = Input text >> Output parse as sequence
    - Code generation = Natural Language >> Python Code

> **seq2seq의 학습 과정 : E2E**

- seq2seq는 Conditional Language Model의 한 종류이다.
- seq2seq의 Decoder는 target sequence에서의 앞선 문맥과 input text가 함께 주어졌을 때의 조건부 확률을 바탕으로 target sequence y의 **다음 단어** 를 예측하기 때문이다. <br/>
<img src = "../../Figures/Lecture 8/Eq 2.svg" width = "500dp">
- P(y|x)를 직접 구하는 것이 아니라 조건부 확률들의 곱으로 분해한 뒤 각 조건부 확률을 구함
- Training 과정에서는 이전 Decoding RNN cell에서의 output을 다음 cell의 input으로 활용하는 것이 아니라, 정답(target sentence in corput)을 input으로 활용해서 weight를 optimize한다.
- 대신 Inference 과정에서는 이전 cell output을 다음 RNN cell input으로 활용한다.<br/>
<img src = "../../Figures/Lecture 8/Fig 3.jpg" width = "500dp"> <br/>

> **Greedy Decoding과 한계점**

- Greedy Decoding은 Local Best의 조합이 Global Best로 이어진다는 아이디어이다.
- Greedy Decoding을 수행하게 되면 Decoder에서 우연히 한 번 잘못된 output을 도출하게 되면 다음 output 역시 연속해서 틀리게 된다.
- 따라서 전체 조합을 모두 순회하여 가장 정확한 output을 도출하는 Exhausive search Decoding 방법론이 있다.
    - O(V^t)의 복잡도를 가지기 때문에 너무 expensive한 방법론이다.
- 그래서 효율적으로 가능성이 높은 몇 가지만 탐색하는 *Beam search Decoding* 방법론이 있다.

> **Beam search Decoding**

- Decoding step에서 가장 가능한 **k개의 partial translation**을 도출, **k개의 hypothesis**를 도출한다고도 함
    - k = beam size(5~10 정도)<br/>
<img src = "../../Figures/Lecture 8/Fig 4.jpg" width = "500dp"> <br/>
- Beam search Decoding은 항상 'optimal'한 decoding 결과를 도출한다고는 장담할 수 없다.
- 하지만 Exhausted search Decoding에 비해 훨씬 효율적인 방법론이다.
- Greedy Decoding에서는 model이 **END** 토큰을 도출할 때까지 계속 decode해야 한다.
    - 하지만 Beam Search Decoding에서는 different timestep에서 **END** 토큰이 도출될 수 있다. 
    - 이때 일찍 **END** 토큰이 반환된 경우 해당 hypthesis는 **종결** 되었다고 보고, 다른 Hypothesis를 계속해서 탐색한다.
    - Beam Search의 종결 조건은 특정 **timestep T** 에 도달하거나 **n개의(predefiened cutoff T, n) 종결된 hypothesis** 가 도출되는 경우이다.

> **Machine Translation 모델의 성능평가, BLEU**

- BLEU(BiLingual Evaluation Understudy)
    - Machine-written translation과 human-written translation을 비교하여 similarity score를 계산함
    - n-gram(1,2,3,4-gram)의 precision을 계산함
    - 너무 짧은 system translation의 경우 페널티를 부과함
- BLEU는 유용하지만, 완벽하지 않은 지표
    - 동일한 문장이라도 번역할 수 있는 방법에 여러 가지 valid translation이 있다.
    - 따라서 good translation이라도 낮은 BLEU score를 가질 수 있다.

> **NMT의 장단점, 여전히 해결하지 못하는 문제들**

- NMT의 장점
    - 더 성능이 뛰어나고, '문맥' 정보를 더욱 잘 반영한다.
    - 하나의 Neural Network가 End-to-End로 optimize된다.
    - 독립적으로  optimize되어야 하는 Subcomponent가 없음
    - Human Engineering Effort가 적게 든다 : Feature Engineering 필요 없고, 모든 언어 쌍에 대해 적용 가능
- NMT의 단점
    - NMT는 디버깅이 어렵고, interpretation이 어렵다.
    - 또한 NMT는 control하기 어렵다 : 예를 들어 특정 혐오단어를 사용하지 말자는 식의 rule, guideline을 추가하는 것이 SMT에서는 쉽지만 NMT에서는 어렵다.

> **Machine Translation의 근본적 한계**

- Out-of Vocabulary Word를 생성할 수 없음
- Domian Mismatch(Train과 Test 데이터 사이의)
- 긴 텍스트에 걸쳐 context를 유지하는 것이 어려움
- Low-resource language pair에서는 학습 data의 bias에 의한 잘못된 결과가 도출되기도 함
    - 말리어로 아무 단어나 쓰면 성경이 튀어나옴
- Common sense를 학습하는 것은 여전히 어려움
- Input Data의 Bias로 인한 편향된 결과 도출
    - 성별과 무관한 문장이어도 성별이 매핑되어 나옴
- 근본적으로 Uninterable System이기 때문에 발생하는 현상

---

### 8.3. Neural Technique : Attention

> **Bottleneck Problem : Seq2Seq의 문제점**

- Encoder의 마지막 단에서 모든 문맥 정보를 캡처하도록 되어 있으므로 병목문제 발생
- 디코더의 각 단계에서, 인코더에 직접 연결해 특정 부분마다 '집중'하도록 하는 작업이 필요
- Attention의 Motivation이 됨

> **Attention Architecture**

<img src = "../../Figures/Lecture 8/Fig 5.jpg" width = "500dp">
- Attention Score를 디코더 벡터와 인코더의 각 벡터들과의 유사도를 활용해 평가하고(점곱, 가중치곱, 등등..), 그 유사도 점수를 가중치로 하여 Attention Output을 도출함
- Attention Output과 Decoder의 state를 concatenate해서 non-attention seq2seq model에 입력하여 학습하는 방식 <br/>
<img src = "../../Figures/Lecture 8/Fig 6.jpg" width = "450dp"><br/>
- Attention은 NMT 성능 향상에 크게 도움이 되고, 이전 인코더의 특정 부분을 '집중(Attention)'하므로 bottlencek 문제를 해결할 수 있음
- 일종의 Shortcut을 구성하므로 Vanishing Gradient에도 도움이 됨
- Attention Score를 바탕으로 한 Attention Distribution을 구성하여 Decoder가 어느 부분에 집중하는지 시각화, Interpretability 향상
- 일종의 Soft alignment를 도출하게 됨

> **General Attention**

- Attention은 NLP Task뿐만 아니라 다른 task로 일반화할 수 있음
- Attention의 일반적인 의미:
    - **벡터 Values, 벡터 Query** 가 주어질 때 **Attention 테크닉** 은 *Values들의 가중치 값을 Query에 의존하도록 설정하여 계산하는 것*이다.
    - 이때 **query** *attends* to the **values** 라고 부른다.
- Attention 과정의 의의는, weighted sum을 수행하게 되면 query에 따라서 어느 value에 집중할지를 선택하게 되고 이는 value에 담겨 있는 정보를 선택적으로 요악하는 과정이 된다.
- 임의 개수의 Representation으로부터(Values), 다른 Representation(Query)에 의존적이도록 고정된 길이의 Representation을 구성하는 방법이기도 하다.
- Attention의 다양한 변형
    - Basic Dot-product Attention : 점곱 연산
    - Multiplicative Attention : 중간에 가중치 Matrix W를 곱하는 내적 연산 정의
    - Additive Attention : 서로 다른 가중치 Matrix W1, W2를 설정하여 디코더의 state와 어텐션 벡터의 크기를 d3로 같게 맞추어 주는 방법. d3는 새로운 하이퍼파라미터가 됨.