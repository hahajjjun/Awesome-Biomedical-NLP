# Lecture 5

### Dependency Parsing

---

### Overview : Linguistic Structure와 Dependency Parsing

|Index|Subtitle|
|--- | --- |
|5.1.| Syntactic Structure : Consistency and Dependency |
|5.2.| Dependency Grammar and Treebanks |
|5.3.| Transition-based dependency parsing |
|5.4.| Neural dependency parsing |

---
### 5.1. Syntactic Structure : Consistency and Dependency

- 언어를 정확하게 해석하기 위해서는 문장 구조를 이해하는 것이 필요하다 : 두 종류의 Ambiguity가 존재한다.
- Phrase Attachment Ambiguity
    - 형용사구(AP), 동사구(VP), 전치사구(PP) 등이 어떤 단어를 수식하느냐에 따라 의미가 달라짐 : 중의적인 의미를 가지기도 한다.
    - **"Scientists observe whales from space"**
        - 과학자들이 우주에서 고래를 관측한다.
        - 과학자들이 우주에서 온 고래를 관측한다. <br/>
        <img src = "..\..\Figures\Lecture 5\Fig 1.jpg" width = "300dp"/> <br/>
- Coordination Scope Ambiguity
    - 특정 단어가 수식하는 범위가 달라지면 이미가 변하는 경우
    - **"Shuttle veteran and longtime NASA executive Fred Gregory appointed to board"**
        - NASA 임원 Fred Gregory와 우주선 베테랑이 이사로 임명되다.
        - NASA 임원이자 우주선 베테랑인 Fred Gregory가 이사로 임명되다.
- 언어학자들이 문장의 구조에 대해서 가지는 관점은 크게 두 가지이다.
    - Phrase Structure Grammars
    - Dependency Structure Grammars
- *Phrase Structure Grammars (Constituency = Context-free Grammars(CFGs))*
    - 문장은 nested된 unit들로 구성되어 있다.
    - 최소 단위인 **Words** 가 모여서 **Phrase**를 만들고, **Phrase**는 더 큰 규모의 **Bigger Phrase**로 조직화된다.
    - *[the, cat, cuddly, by, door]* 의 단어들은 category [명사, 동사, 형용사, 부사...]로 분류될 수 있다.
    - 이렇게 categorize된 단어들이 만드는 phrase 역시 categorical하게 분류할 수 있고, 이 과정은 재귀적으로 이루어진다.
    - ['the', 'a'] : *Determiners, Det*
    - ['cat', 'dog'] : *Nouns, N*
    - ['the cat', 'a dog'] : *Noun Phrase, NP, Det N (Determiners followed by nouns)*
        - Noun Phrase는 Det + (Adj, 필요하다면) + N + PP(prepositional phrase) 으로 구성될 수 있다.
        - Prepositional phrase(PP)는 다시 Prep + NP로 구성될 수 있다.
        - NP ⇒ Det (Adj) N PP
        - PP ⇒ Prep(P) NP
        - VP ⇒ V P
        - 이렇게 주어진 문장을 unit word들의 category로 tagging할 수 있다. <br/>
            <img src = "..\..\Figures\Lecture 5\Fig 2.jpg" width = "300dp" />
- Dependency Structure Grammars
    - 한 단어가 다른 단어를 수식하는 관계에 따라 문장을 구조화
    - 수식하는 관계는 품사와 무관함
    - 수식하는 단어를 'head', 'governor', 수식받는 단어를 'dependent', 'modifier'라고 함 <br/>
        <img src = "..\..\Figures\Lecture 5\Fig 3.jpg" width = "300dp"/>
    
    - Dependency syntax는 문법적 구조가 lexical items 사이의 연관성을 포함하고 있다고 가정하고, 주로 화살표로 dependency를 표현한다.
    - nmod(noun modifier),  nsubj(noun subject), appos(apposition), aux(auxillary) 등등의 태그로 dependency 관계의 type을 표시할 수도 있다. <br/>
    - 우선적으로 type를 배제한 경우만 살펴볼 것이다.

### 5.2. Dependency Grammars and Treebanks
- Dependency Grammar
    - 가장 앞에 fake ROOT unit을 덧붙여 항상 모든 성분들은 최종적으로 ROOT를 수식하도록 한다.
    - Sequence Dependency structure 
        - Arrow는 **dependent**(수식받는 단어) ⇒ **head** (수식하는 단어)방향으로 그린다.
        - Dependency 관계의 type 역시 함께 표기한다.
        - ROOT에 의해 수식받는 단어(dependent)는 유일해야 한다.
        - 순환 cycle이 생기면 안 된다.
        - arrow 끼리 교차하는 경우 : extra modifier의 순서를 바꿔서 제거할 수 있다. <br/>
        <img src = "..\..\Figures\Lecture 5\Fig 4.jpg" width = "300dp"/>
    - Tree Dependency structure
        - Sequence를 Tree로 변환할 수 있다.
        - 가장 상위 노드가 ROOT, 하위 노드가 상위 노드를 항상 수식하는 방향으로 트리를 구성한다. <br/>
            <img src = "..\..\Figures\Lecture 5\Fig 5.PNG" width = "300dp"/>
- Dependency parsing에서 주로 관찰되는 특징
    - Bilexical Affinities
        - 두 단어 사이의 친화성(호응)에 따라 plausible한 연결을 탐색
    - Dependency Distance
        - 가깝게 위치한 단어일수록 수식 관계에 있을 가능성이 높음
    - Intervening material
        - intervening verb나, 구두점을 넘나들어 수식 관계가 형성되지 않는 경향을 가짐
    - Valency of heads
        - 단어마다 자신이 head가 되었을 때 좌우에 dependents가 몇 개 정도 분포하는지 특성이 있음
- *Marcus et al. 1993*, **Universal Dependencies treebanks**
    - Treebank : corpus annotation
    - 말뭉치(corpus)의 본문에 dependency type별로 annotation을 수행해 놓은 database
    - Reusable
    - 여러 Language corpus에 대한 annotation <br/>
        <img src = "..\..\Figures\Lecture 5\Fig 6.PNG" width = "600dp"/> <br/>

---

### 5.3. Transition-Based Dependency Parsing
- Methods of Dependency Parsing
    - 동적 프로그래밍
        - complexity O(n^3)
    - 그래프 알고리즘 
        - create Minimum Spanning Tree for a sentence
    - Constraint Satisfaction
        - Hard Constraints를 만족하지 못하는 edge들을 제거하는 방식
    - **Transition-based parsing**
- Transition-based parsing(Deterministic dependency parsing)
    - Greedy 'choice' of attachments
    - 'choice'는 ML classifier로 수행함
- *Nivre et al. 2008*, **Algorithms for Deterministic Incremental Dependency Parsing**
    - Greedy transition-based parsing
    - Parser는 Bottom-up actions을 수행함
        - stack, buffer, set of arcs, action 으로 구성
        <img src = "..\..\Figures\Lecture 5\Fig 7.jpg" width = "600dp"/> </br>
        - stack에 저장된 token으로 state를 classify : softmax clsassifier 같은 discriminative classifier 사용
        
    - State를 ML classifier input으로 제공하기 위한 state feature embedding 방법론(*Nivre and Hall. 2005*, **MaltParser**)
        - 토큰 묶음의 형태로 존재하는 state를 embedding하기 위해서 토큰의 feature notation, 특히 토큰의 tag(POS tag)를 참고하기도 함
        - 발생할 수 있는 모든 notation들의 조합을 상정하고, 해당 조합이 있으면 1로 binary embedding 수행하는 방식(Sparse Encoding)
            - 모든 notation의 조합을 feature template라고 부름
            - feature template는 대략 1e+6 ~ 1e+7개 정도 존재하며, 1~3개 정도의 요소 조합으로 구성됨
            - Feature를 binary encoding하여 계산할 때 연산 비용이 많이 필요
            - POS tag 및 여러 태그의 의미론적 차이 반영 X
                <img src = "..\..\Figures\Lecture 5\Fig 8.jpg" width = "600dp"/> </br>
    - Parser 성능평가
        - 일반적인 Accuracy
        - UAS(Unlabeled Attachment Score) : Dependency 관계의 정답 여부만 확인, 관계의 종류는 무시
        - LAS(Labeled Attachment Score) ; Dependency 관계 + classification까지 정확해야 정답으로 인정
            <img src = "..\..\Figures\Lecture 5\Fig 9.jpg" width = "600dp"/> </br>
        
---

### 5.4. Neural Dependency Parsing
- Neural Network가 적용된 Dependency Parsing 방법론
- 왜 Neural Dependency Parsing : Transition based 방법론의 문제점
    - Feature Embedding 수행해도 굉장히 sparse
    - 불완전한 Feature Embedding
    - Embedded Feature를 활용한 계산 비용이 많이 드는 문제(Parsing time의 95%를 차지)
- Transitional, Graph-Based, Neural Parsing model 성능비교
    |Parser|UAS|LAS|sentence/sec|-|
    |-|-|-|-|-|
    |MaltParser|89.8|87.2|469|Transitoinal|
    |MSTParser|91.4|88.1|*10*|Graph-Based|
    |TurboParser|**92.3**|89.6|*8*|Graph-Based|
    |Chen and Manning, 2014|92.0|**89.6**|**654**|Neural|
- Input layer : Input state representation <br/>
    <img src = "..\..\Figures\Lecture 5\Fig 10.jpg" width = "600dp"/> <br/>
- Hidden layer : Cube activation function(ReLU, LeakyReLU, Sigmoid, Tanh ...)
- Output layer : Softmax를 통해 Decision-making
- Structure <br/>
<img src = "..\..\Figures\Lecture 5\Fig 11.jpg" width = "600dp"/> <br/>
- MaltParser(Nivre et al.)에서의 sparse representation를 dense representation으로 보완
    - accuracy, speed 측면에서도 모두 outperforming
    - bigger, deeper model + hyperparemter tuning을 통해서 계속해서 성능 개선이 이루어짐(Weiss et al.(2015), Andor et al.(2016))
        
