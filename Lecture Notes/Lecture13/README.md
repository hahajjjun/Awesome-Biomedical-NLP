# Lecture 13

### Contextual Word Embeddings

---

### Overview : 문맥 속에서의 단어 임베딩

|Index|Subtitle|
|--- | --- |
|13.1.| Reflection on word representations |
|13.2.| Pre-ELMo and ELMo |
|13.3.| ULMfit and onward |
|13.4.| Transformer architectures |
|13.5.| BERT |

---

### 13.1. Reflection on word representations

> **Pre-trained word vectors: The early years(2011)**

- 단어의 representation을 학습하는 것은 굉장히 중요한 Task입니다. 지금까지 우리가 배웠던 word vector로는 *Word2vec, GloVe, fastText* 등이 있었습니다.
- 각 단어를 일반적으로 하나의 representation으로 대응시키는 representation learning은, 개념적으로 생각해 보면 거대한 corpus만을 활용해 학습하는 unsupervised learning task입니다.
- 그리고 이러한 pre-trained word vector가 있었기 때문에 Neural Network 모델들이 NLP task에 있어서 좋은 성능을 내기 시작할 수 있었습니다. 
실제로 BiDAF 아키텍처나 Attentive Reader Model이 아니더라도, 심플한 RNN, LSTM의 input으로도 pre-train되어 있는 GloVe Embedding을 활용하는 경우가 많았습니다.


- 2011년까지만 해도, POS(Part of Speech) Tagging 그리고 NER(Named Entity Recognition) Task에 대한 성능의 Landscape는 다음과 같습니다.

<p align = "center"><img src = "../../Figures/Lecture13/Fig 1.JPG" width = "500dp"></p>

- 특징을 살펴보면 다음과 같습니다.
    1. SOTA는 Classical한 Feature-based model이었습니다.
    2. Supervised Neural Network는 POS Tagging Task에서는 97% accuracy 내외로, SOTA 모델 못지않은 성능을 내고 있었지만 NER Task에서는 SOTA 모델에 약 8%나 떨어지는 F1-score를 보이며 좋은 퍼포먼스를 보이진 못했습니다. 
    3. "동일한 데이터셋만"을 주고서 학습을 하면 Categorical Feature-based model(CRF, SVM ...)에 비해서 NN은 성능이 떨어졌습니다.
    4. Neural Network를 활용한 NLP Task가 인기를 끌기 시작한 것은 Unsuperviesd pre-training 결과를 바탕으로 supervised Neural Network를 train하기 시작하면서부터입니다.
    5. Neural Network는 "동일한 데이터셋"이라 할지라도 더욱 큰 Vocabulary로부터 unsupervised 방식으로 이전에 학습된 정보를 가져올 수 있었기 때문입니다. 2014년 이후부터 Neural Network는 POS, NER task 모두에서 SOTA를 찍기 시작했습니다.

- 그리고 Unsuperviesd Word Embedding 과정에서 발생할 수 있는 가장 일반적인 문제가 **Unkown Words Problem**입니다.
    - 가장 보편적으로는 5번 미만 등장하는 rare words를 UNK로 동일하게 취급하여 UNK에 대한 word vector를 학습하는 것입니다. OOV(Out of Vocab) word를 모두 UNK로 취급해 train하는 것입니다.
    - 이 방법은 UNK word끼리 구분할 수 없다는 문제점이 있었고, 그래서 해결책으로 알파벳(character) 단위로 model을 설계하여 vector를 학습하기도 합니다.
    - 또다른 해결책으로는 UNK word가 등장할 때마다 random vector를 assign하고 vocab에 추가하는 방식도 있습니다. 
    - 마지막으로 UNK word들을 최소한의 몇 개 클래스로 분류해 클래스 정보만 추가해 줄 수도 있습니다.

> **Problem of single-word embedding**

- 각 단어마다 하나의 임베딩 벡터를 추가하는 것이 가지는 치명적인 문제점이 있습니다.
    - 첫째는 모든 단어는 뉘앙스와 문맥에 따라서 다른 의미를 가질 수 있기 때문입니다. 굉장히 fine-grained word sense를 가진 단어의 경우에는 그 의미가 모호하기도 해서, 하나의 벡터로 학습하는 것이 본질적으로 어려울 수 있습니다.
    - 둘째로는 의미적으로는 비슷하지만 의미적/문법적 용법의 차이로 다르게 인식되는 단어들이 있기 때문입니다. 'arrive'와 'arrival'의 경우에는 의미적으로 비슷하지만, 그 용법이 다르기 때문에 전혀 다른 벡터로 임베딩될 수 있습니다.
- NLM(Neural Language Model)에서도 봤듯이, LSTM, RNN 모델은 이전 단어의 문맥을 활용해 다음 단어를 예측했습니다. 따라서 'arrive'와 'arrival'은 의미적으로는 같지만 문법적 차이로 인해 주위에 오는 단어들이 달라지게 되고, 이로 인해 모델은 두 단어를 전혀 다른 벡터로 임베딩 할 수도 있을 것입니다.
<p align = 'center'>
요약하자면, 지금까지 배웠던 LM들은 context-specific word representation을 생성하고 있습니다.
</p>

---

### 13.2. pre-ELMo and ELMO

> [pre-ELMo : TagLM, Peters et al.(2017)](https://arxiv.org/pdf/1705.00108.pdf)

- 개념적으로 Word2Vec이나 GloVe 등의 임베딩 방법론으로는 Contextual Word Embedding이 불가능합니다. 그래서 Language Model을 임베딩에 활용한 아이디어가 TagLM에서 제시되었습니다.
- TagLM은 pretrain된 Bilinear Language Model을 활용해 Sequence Tagging Module의 임베딩 과정에서 추가 임베딩을 수행하게 됩니다.
- 전체 모델의 학습은 Semi-Supervised fashion으로 이루어지게 됩니다.
<p align = "center"><img src = "../../Figures/Lecture13/Fig 2.jpg" width = "600dp"></p>

- Bi-LM의 경우에는 고정된 임베딩을 제공하고, Backpropagation 과정에서 gradient flow가 일어나지 않도록 고정하는 것이 특징입니다.
- Bilinear LM은 Bi-LSTM과 개념적으로 다른 것입니다! *Bi-LSTM은 순방향 LSTM과 역방향 LSTM의 은닉 상태를 concatenate하여 최종적으로 산출한 결과를 다음 층의 입력으로 사용하지만, Bi-LM은 순방향 언어모델과 역방향 언어모델의 별개 언어모델을 학습하는 개념입니다.*
- 여기서 Language Moodel이 single layer라면 Bi-LSTM에서 단순하게 concat하는 것과 결과가 동일하겠지만, 뒤에 설명할 ELMo의 경우 Bi-LM이 두 층으로 구성되어 있고, 각각의 LM을 활용해 임베딩 벡터를 얻는 과정은 Bi-LSTM의 simple concatenation과는 다릅니다.
<p align = "center"><img src = "../../Figures/Lecture13/Fig 3.jpg" width = "450dp"></p>

> **ELMo, Embeddings from Language Model**

- ELMo는 TagLM의 Semi-Supervised Fashion Training 그리고 Bilinear Language Model 부분을 활용하면서 디테일하게 여러 부분에서의 성능 향상을 이루어낸 모델입니다.
- ELMo는 임베딩 과정의 단순화와 차원 축소를 통해서 파라미터 수를 줄인 것이 특징이며, 비교적 경량화된 Language Model을 사용하고 있는 모델입니다.
- Bidirectional LM을 학습한다는 점은 동일하고, 대신 Initial Word Representation 과정에서 Word-level embedding은 수행하지 않으며 **Char-CNN**만으로 word representation을 구축합니다.
- Char-CNN에서는 2048개의 필터와 2개의 highway layer를 사용하는데, output을 다시 feed-forward layer를 통해 512 dimension으로 projection하여 차원축소를 하고 있습니다.
- 4096 dimension의 hidden 그리고 cell state를 가지는 LSTM을 활용하고 있고, 다음 input 과정에서도 512 dimension으로 다시 차원을 축소합니다.
- *Residual Connection*과 *Parameter Tying*이라는 테크닉도 활용되었습니다. 
- ELMo에서 제안한 여러 층으로 구성된 Bi-LM으로부터 임베딩을 얻는 과정은 다음과 같고, 이러한 임베딩을 ELMo Embedding이라고 부릅니다.
- ELMo 임베딩은 Contextual한 관점에서 단어의 의미를 Capture할 수 있다는 장점이 있어, 기존 Baseline 모델에 ELMo 임베딩을 추가했을 때에도 성능 향상이 이루어지는 것이 보고되었습니다.

<p align = "center"><img src = "../../Figures/Lecture13/Fig 4.jpg" width = "450dp"></p>

- ELMo 임베딩을 수행하는 Bi-LM은 두 개의 레이어로 구성되어 있다고 언급했는데, 그 중에서 Lower Layer와 Higher Layer가 capture하는 정보가 달랐습니다.
- Lower Layer는 lower-level syntax, 예를 들어 POS tagging, Syntactic depedencies, NER 등을 잘 capture했습니다.
- 반면에 Higher Layer는 higher-level semantics, 예를 들어 Sentiment, Semantic role labeling, question answering, SNLI 등을 잘 수행했습니다.
- 만약 여기서 Bi-LM의 레이어가 두 개 이상인 경우에는 어떻게 되는지 역시 흥미로운 연구주제입니다.

---

### 13.3. ULMfit and onward

- ULMfit
    - General한 도메인 Corpus을 활용해 LM을 pre-train하고, 타깃 도메인의 데이터를 활용해 LM을 fine-tuning하게 됩니다.
    - 그리고 Bi-LM 모듈의 마지막 레이어만 classification layer로 바꾸고, 원래 LM의 마지막 레이어의 파라미터는 Freeze합니다.
    - LM fine-tuning 과정에서는 여러 디테일한 Learning Rate 조절 기법들이 활용되었습니다.(STLR, 층마다 다른 LR 적용..)

<p align = "center"><img src = "../../Figures/Lecture13/Fig 5.jpg" width = "600dp"></p>

- ULMfit 이후로 모델들은 주로 Scale-UP의 형태로 발전해 왔습니다.
- OpenAI의 GPT는 1 GPU day의 ULMfit보다 240배 더 긴 시간동안 Train했고, BERT 역시 그러합니다. 최근에는 더욱 큰 모델이 더욱 좋은 성능을 내고 있음이 보고되고 있습니다. 
- 그리고 이러한 모델들은 전부 Transformer 계열의 모델입니다!

---

### 13.4. Transformer Architectures

- 이전의 RNN에서 Attention을 활용해 특정 부분에 '집중'하도록 하는 아키텍처가 있었습니다.
- RNN은 본질적으로 parallelization이 불가능했고, 그래서 Attention만을 활용해 좋은 성능을 낼 수는 없을지 고민하게 되었습니다.
- Attention is all you need, a.k.a Transformer는 이러한 배경 속에서 제안된 복잡한 Attention 기반 모델입니다.
- [Transformer Pytorch Implementation](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

> **Transformer**

- Attention Concept : Key, Query, Value
- Positional Embedding
- Encoder Self-Attention
- Decoder masked Self-Attention : '미리보기'를 방지하는 마스크로, look-ahead mask라고도 부릅니다. 현재 시점의 예측에서 미래에 있는 단어들은 masking을 통해 참고하지 못하도록 했습니다.
- Encoder - Decoder Attention
- Skip Connection 등의 기법이 적용되었습니다.
- FFNN : Position-wise Feed forward neural network를 사용합니다.(Position마다 각각 독립적인 FFNN이 사용됩니다)

<p align = "center"><img src = "../../Figures/Lecture13/Fig 6.jpg" width = "900dp"></p>

<p align = "center"><img src = "../../Figures/Lecture13/Fig 7.jpg" width = "900dp"></p>

---

### 13.5. BERT
