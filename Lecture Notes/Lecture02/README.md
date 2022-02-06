# Lecture 2

### Word Vectors and Word Senses

---

### Overview : 여러 가지 단어 임베딩 모델

|Index|Subtitle|
|--- | --- |
|2.1.| Word2Vec recap |
|2.2.| Word2Vec algorithm family |
|2.3.| LinAlg based models : counting |
|2.4.| GloVe model |
|2.5.| Intrinsic/Extrinsic Evaluation of word vectors |
|2.6.| Word senses |

---

### 2.1. Word2Vec recap

- Word Vector는 **랜덤하게 초기화**된 상태
- 전체 **corpus를 순회**함, sliding window
- Model은 **center word vector ⇒ outer context word vector를 추정**하는 일을 수행
- 그 과정에서 optimization되는 parameter = word vector embedding에 해당
- 학습 과정 = word vector를 update해서 surrounding word를 더 잘 추정할 수 있도록 하는 과정
- P(o|c) = softmax(dot(uo, vc))이므로 **matrix notation**으로도 표현 가능

<aside>
💡 Why?

Word Vector Embedding 결과를 2D, 3D Projection해서 clustering했을 때 유사한 단어끼리 비슷한 위치에 존재하는 이유

⇒  단어가 의미적으로 호응하지 않으면 같은 window 내 등장이 적어 P(o|c)를 작게 추정해야 하므로

⇒  두 word vector의 dot product가 감소함(projection해도 서로 다른 방향을 가리키게 됨)

</aside>

- 단어의 순서와 위치는 고려하지 않고, 단어의 출현 빈도에 의존 → **Bag of Words model과도 유사함**
- Optimization : Gradient Descent (single update할 때도 연산이 많이 필요함)
    - Stochastic Gradient Descent을 통해서 sample window 혹은 small batch에 대해서 update시행
    - Very Sparse Gradient Vector

---

### 2.2. Word2Vec algorithm family

- **Word Vector Embedding 개수에 따른 분류**
    - 지금까지 설명한 Word2Vec 모델은 **한 단어당 두 개의 word vector**(center, outside(context))가 임베딩함 ⇒ 최종 word vector 산출 시에는 두 word vector의 평균으로 구하게 됨
        - 직관적이고 단순한 방법
    - **한 단어당 한 개의 word vector embedding**을 수행하는 variated model
        - slightly better performance
        - much complicated calculus of gradient
- **Loss function을 구성하는 방식에 따른 분류**
    - Skip-Grams(SG)
        - center word가 주어지면 ⇒ outside word를 예측하는 순서 : P(o|c)
    - Continuous Bag of Words(CBOS)
        - context words를 바탕으로 ⇒ center word를 에측하는 순서 : P(c|o)
- **그 외 Additional Variations**
    - Negative Sampling
        - Naive Softmax 개선 ⇒ Loss function에 logistic term 추가
        - **아이디어 : 좋은 모델이라면 true pair(c,o)와 random noise(c,o)를 구분할 수 있어야 한다!**
            
<p align = "center">
<img src = "../../Figures/Lecture02/Fig 1.png" width = "600dp"/>
</p>
            

---

### 2.3. LinAlg based models : counting

- ***COALS; Rohde, Gonnerman, & Plaut, 2005***
    - Co-occurence matrix를 구성 ⇒ SVD 같은 LinAlg based calculation 활용
    - **Co-occurence matrix building 방법**
        - **Sliding windows**
            - Word2Vec과 유사, 각 단어마다 window를 설정해 slide
            - **syntactic, semantic information을 capture할 수 있음**
                - syntactic 예시 : fly ⇒ flying : eat ⇒ ?
                - semantic 예시 : man ⇒ king : woman ⇒ ?
        - **Full document**
            - 전체 doc에 대한 co-occurence matrix는 전체 text의 general한 topic을 알려줌
            - **Latent Semantic Analysis(LSA)를 수행할 수 있음**
        - Co-occurence matrix 전체를 사용할 수도 있음 : But very high dimensional & Sparse
            - less robust model & sparsity issue
            - **SVD 통한 차원 축소**

<p align = "center">
<img src = "../../Figures/Lecture02/Fig 2.png" width = "400dp"/>
</p>

- Co-occurence matrix X를 구성할 때 사용한 Hacks
    - Raw count에 대해서 SVD를 수행하면 사실 성능이 좋지 않음
    - Scaling the counts
        - 빈도가 너무 높은 function words제외
        - 빈도 = max(X,100)으로 threshold하기
        - log(빈도)로 계산
    - 단순히 count 쓰기보다 Pearson correlation 사용해 가중치를 다르게(negative vals ⇒ 0)
    - 중심에서 가깝게 나타날수록 더 높은 가중치 : Ramped Windows(🔺)

---

### 2.4. GloVe Model

- **지금가지 소개한 모델들 분류해서 정리한 결과**
    
    
    | Count Based | Direct Prediction |
    | --- | --- |
    | LSA(Latent Semantic Analaysis), HAL | Skip-Gram & CBOW |
    | COALS, Hellinger-PCA | NNLM, HLBL, RNN |
    | 장점 1 : Fast Training | 장점 1 : Improved performance |
    | 장점 2 : Efficient Usage of statistics | 장점 2 : Capture complex patterns |
    | 단점 1 : 단순한 Word similarity 정도만 capture 가능 | 단점 1 : Corpus Size에 따라 학습 속도 느려짐 |
    | 단점 2 : 빈도가 높게 측정된 단어에 importance 불균등화 | 단점 2 : Inefficient Usage of Statistics |
- ***Encoding meaning components in vector differences [Pennington et al, EMNLP 2014]***
    - Co-occurence probability의 비율이 의미적 유사성을 인코딩할 것이라고 추론함
        
<p align = "center">
<img src = "../../Figures/Lecture02/Fig 3.png" width = "400dp"/>
</p>

- Counting 통해서 embedding vector 게산하거나 유추 가능
        
<p align = "center">
<img src = "../../Figures/Lecture02/Fig 4.png" width = "300dp"/>
</p>
        
- ***GloVe [Pennington et al, EMNLP 2014]***
    <p align = "center">
    <img src = "../../Figures/Lecture02/Fig 5.png" width = "400dp"/>
    </p>

    - Fast Training / Scalability가 장점
    - 작은 규모의 Corpus에서도 좋은 성능을 보임

---

### 2.5. Intrinsic/Extrinsic evaluation of word vectors
- Intrinsic
    - 전체 model의 task중 일부분인 subtask에 대해서 performace measure
    - 빠르고, 비용이 적게 듦
    - Real task와의 correlation을 살펴보는 데에는 유용하지 않을 수 있음
- Extrinsic
    - 전체 Real task에 대해 evaluate
    - 각각의 subtask중 어느 부분에서 문제가 있는지, 아니면 subtask system의 interaction에 의한 것인지 등등  예상할 수 없음
    - 시간이 많이 걸리고, 비용이 많이 듦
- Example of intrinsic word vector evaluation
    - Word vector analogy
        - syntactic / semantic analogy를 평가
        - analogy task를 잘 수행하면 robust한 embedding이라고 주장
        - dimensionality, window size 등의 hyperparameter를 변경시키면서 analogy task의 정확도를 비교함
        - training time, corpus source 등에 따라서도 human knowledge와의 correlation이 달라짐
    - Human Judgement
        - Psychological
        - Human Scoring -> 평균을 구해서 사용
- Example of extrinsic word vector evaluation
    - 모든 subsequent task를 수행한 결과에 대한 evaluation
    - named entity recognition(개체명 인식, NER) : 이름을 가진 개체를 인식
    - word2vec embedding 이후에 classify하는 과정이 sequentail하게 이루어짐
      
---

### Word Senses
- Ambiguity of word sense
- 대부분의 단어는 여러 의미를 가지고 있음
- common words
- one vector만으로 모든 meaning을 capture할 수 있을까?
- *Huang et al. 2012*, **Improving word representations via global context and multiple word prototypes**
    - word window 내의 단어들을 클러스터링
    - 같은 단어도 여러 클러스터에 속할 수 있음
- *Arora et al. TACL 2018*, **Linear Algebraic Structure of Word Senses, with Applications to Polysemy**
    - Word2Vec 같은 word embedding을 통해 얻어진 word vector의 superposition으로 최종 word vector를 도출함
    - 선형 결합의 가중치는 해당 의미로 쓰여진 frequency의 상대적 abundance로 결정
    - 직관적으로는, superposition을 하면 정보가 소실되는 것 아닌가?
        - high-dimension word vector는 sparse하게 coding되어 있음
        - 그래서 통계적인 방법론을 활용해 re-seperate하는 것이 가능하다고 주장
        - [0.5, 0.0, 0.5, -1.0, -1.0, 0, 0] >> AVG([1.0, 0, 1.0, 0, 0, 0], [0, 0, 0, -2.0, -2.0, 0, 0]
