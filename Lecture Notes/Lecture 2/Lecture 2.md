# Lecture 2

### Neural Classifiers

---

### Overview : 여러 가지 단어 임베딩 모델

| Word2Vec recap |
| --- |
| Word2Vec algorithm family |
| LinAlg based models : counting |
| GloVe model |
| Intrinsic/Extrinsic Evaluation of word vectors |
| Word senses |

---

### Word2Vec recap

- Word Vector는 **랜덤하게 초기화**된 상태
- 전체 **corpus를 순회**함, sliding window
- Model은 **center word vector ⇒ outer context word vector를 추정**하는 일을 수행
- 그 과정에서 optimization되는 parameter = word vector embedding에 해당
- 학습 과정 = word vector를 update해서 surrounding word를 더 잘 추정할 수 있도록 하는 과정
- P(o|c) = softmax(dot(uo, vc))이므로 **matrix notation**으로도 표현 가능

<aside>
💡 ***Why?***

Word Vector Embedding 결과를 2D, 3D Projection해서 clustering했을 때 유사한 단어끼리 비슷한 위치에 존재하는 이유

⇒  단어가 의미적으로 호응하지 않으면 같은 window 내 등장이 적어 P(o|c)를 작게 추정해야 하므로

⇒  두 word vector의 dot product가 감소함(projection해도 서로 다른 방향을 가리키게 됨)

</aside>

- 단어의 순서와 위치는 고려하지 않고, 단어의 출현 빈도에 의존 → **Bag of Words model과도 유사함**
- Optimization : Gradient Descent (single update할 때도 연산이 많이 필요함)
    - Stochastic Gradient Descent을 통해서 sample window 혹은 small batch에 대해서 update시행
    - Very Sparse Gradient Vector

---

### Word2Vec algorithm family

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
            
            ![PNG 이미지.png](Lecture%202%20b4c53a8069914dcd887f9548fbe23868/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5.png)
            

---

---

### LinAlg based models : counting

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
                
                ![PNG 이미지.png](Lecture%202%20b4c53a8069914dcd887f9548fbe23868/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%201.png)
                
    - Co-occurence matrix X를 구성할 때 사용한 Hacks
        - Raw count에 대해서 SVD를 수행하면 사실 성능이 좋지 않음
        - Scaling the counts
            - 빈도가 너무 높은 function words제외
            - 빈도 = max(X,100)으로 threshold하기
            - log(빈도)로 계산
        - 단순히 count 쓰기보다 Pearson correlation 사용해 가중치를 다르게(negative vals ⇒ 0)
        - 중심에서 가깝게 나타날수록 더 높은 가중치 : Ramped Windows(🔺)

---

### GloVe Model

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
        
        ![PNG 이미지.png](Lecture%202%20b4c53a8069914dcd887f9548fbe23868/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%202.png)
        
    - Counting 통해서 embedding vector 게산하거나 유추 가능
        
        ![PNG 이미지.png](Lecture%202%20b4c53a8069914dcd887f9548fbe23868/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%203.png)
        
- ***GloVe [Pennington et al, EMNLP 2014]***
    
    ![PNG 이미지.png](Lecture%202%20b4c53a8069914dcd887f9548fbe23868/PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5%204.png)
    
    - Fast Training / Scalability가 장점
    - 작은 규모의 Corpus에서도 좋은 성능을 보임

---

### Intrinsic/Extrinsic evaluation of word vectors

---

### Word Senses