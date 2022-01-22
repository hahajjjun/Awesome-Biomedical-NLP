# Lecture 3

### Neural Classifiers

---

### Overview

|Index|Subtitle|
|--- | --- |
|3.1.| Classification |
|3.2.| Neural Networks |
|3.3.| Named Entity Recognition(NER) |
|3.4.| True vs. Corrupted word window binary classification |
|3.5.| Matrix Calculus |

---
### 3.1. Classification
- 고정된 2D word vector를 분류
- softmax/logistic classification
    - linear한 decision boundary 때문에 complex task 어려움
    - MLE approach:
        - 이미지 1
    - Cross-Entropy approcah:
        - 이미지 2
        - Cross-Entropy loss & MLE loss are identical
- Complex한 decision boundary를 위한 Neural Net
    - Nonlinear decision boundary & Learn Complex functions
    - NLP에서의 classification 문제가 특이한 점
        - weight W (C X d matrix)를 학습할 뿐만 아니라 word embedding (V X d matrix)을 함께 학습
        - 다시 말해 conventional한 parameter + representation을 함께 학습
        - 이를 종합하면 (C+V) X d의 차원을 가짐
        - V가 굉장히 크므로 parameter representation이 많아지고, grdient 계산 역시 expensive해짐

---

### 3.2. Neural Networks
- Neural Network는 regression을 동시에 여러 차례 수행하는 연산
- linear 연산들의 합성으로는 linear한 연산밖에 생성할 수 없음 : nonlinear function f가 필요(예: sigmoid)
- 층이 많아질수록 복잡한 classification boundary 가능
- 그러나 Deep layer에서는 Gradient Vanishing, Overfitting 문제도 발생함
    - **ReLU activation function** instead of Sigmoid
    - **Dropout** to get rid of overfitting
- 이미지 3

---

### 3.3. Named Entity Recognition(NER)
- NER task는 이름을 **찾는** task와 이름을 **분류**하는 task로 구성되어 있음
- NER task의 활용
    - 특정한 entity를 문서에서 찾거나 추적하는데 활용
    - Question Answering task 과정에서도 필요
    - Slot-filling classification으로 확장 가능
    - Named Entity Linking/Canonicallization을 NER 이후 수행하기도 함
- Context 속의 단어를 clasify해서 entity를 예측 ⇒ word subsequency의 형태로 entity를 추출함
- 개체명 태깅을 위한 ***BIO encoding***
    - B : Begin, 개체명이 시작되는 부분
    - I : Inside, 개체명의 내부 부분
    - O : Outside, 개체명이 아닌 부분
    - 예를 들어, '해리포터를' ⇒ '해 : B-movie', '리 : I-movie', '포 : I-movie', '터 : I-movie', '를 : O' 의 형태로 태깅됨
    - 개체명 태깅 시에는 BIO encoding과 class 분류를 함께 표기함
- NER이 어려운 이유?
    - "고품격 1등 증권사" vs 고품격(인) "1등 증권사"
    - "미래초등학교" vs 미래(의)초등학교
    - "바다"가 가수 이름인지 진짜 바다인지 모름
    - "to sanction"은 "허용하다"와 "처벌하다"의 의미를 동시에 가짐

---

### 3.4. True vs. Corrupted window binary classification

- 위의 Class Ambiguity해결책 : **Window Classification**
    - **아이디어** : Neigboring word와 함께 있는 context window 내에서 단어를 classify
    - 가장 Naive한 접근은, 특정 단어를 중심으로 한 window 내의 word vector를 평균 내어 average vector를 classsify하는 방법
        - position information loss 발생
    - 다른 접근은, window 내 단어들의 word vector를 일렬로 concat해서 (2m+1) X d 만큼의 길이를 가지는 vector 만들어서 classify하는 것
        - 이미지 4

---

### 3.5. Matrix Calculus

- Jacobian Chain Rule을 활용해 직접 계산
    - Jacobian Form이 computing하기에는 쉬움 <br/>
        <img src="latex 1"/>
        <br/>
    - Gradient와 Parameter의 format이 같기 때문
    - 하지만 Stochastic Gradient Descent를 위해서는 column vector의 형태로 output이 되어야 함 : *Shape convention issue*
    - Trade-off between **"Calculus-Convenient Jacobian Form" and "Column Vector form for SGD implmentation"**
- 따라서 Backpropagation을 통해 이 문제를 해결할 수 있음(Lecture 4)


     


