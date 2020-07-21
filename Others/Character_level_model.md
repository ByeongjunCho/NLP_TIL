# Character level models

### 장점

1. 단어가 가진 의미론적 정보를 버림(give up)  => 단점일 수도 있지만 여기서는 장점으로
2. 입력 어휘 문제 완화 => Word tokenizer에 없을 수 있는 오탈자/방언/약어 등에 강점을 가짐
3. 모델 출력 computational bottleneck을 완화



## EX 1. [OpenAI models can understand sentiment ](https://openai.com/blog/unsupervised-sentiment-neuron/)

> mLSTM을 사용해 Amazone reviews를 학습. 감정에 맞는 문장 생성



## EX 2. [Character level models can translate](https://arxiv.org/abs/1610.10099)

> RNN/LSTM 를 사용하지 않고 Character level에서 학습을 수행한 논문



### Character level 사용의 장점

1. Open vocabulary에 있는 단어도 문제없이 처리 가능. Word라면 <unknown>이 될 단어도 처리할 수 있음.
2. 일반적인 NLP 모델은 문장이 들어오면 다음 문장을 예측하는 방식으로 구성되어 있음. 이때 다음 단어를 판단하기 위해서는 `softmax`를 사용하게 됨. Vocab의 개수가 매우 많기 때문에 `softmax`를 계산하면서 많은 computational power를 사용함.



### Character level NLP models의 단점

1. 의미가 사라짐(Semantically void)

   Semantic 정보를 포기하는것은 문제가 있음. Word 단위에서 성능이 좋은 Google BERT 같은 pre-trained 모델이 존재. (Asia languages는 character level에서 BERT구조로 동작이 잘 된다고 한다.)

2. 긴 sequences는 Computational expense를 증가시킬 수 있음

   Character level로 전처리를 하게 되면 처리할 단어의 길이는 `word`와는 다르게 `word 1개당 길이`x`sequence length`로 매우 길어질 수 있음. (입력을 순차적으로 처리하는 RNN/LSTM/GRU 같은 모델에서는 단점이지만 BERT, Transformer같은 병렬처리가 가능하다면 문제가 되지 않음. **해석이 애매함**)

3. Don't care about characters

   Sequence tagging 작업을 하면서 신경쓸 필요가 없다. Character level model을 사용한다는 것은  출력또한 character level로 나오게 됨. 이것은 안좋을 수가 있음. New class에 대한 에러가 생길 수 있음. 해당 문제를 해결하기 위해서는 B-I-O scheme를 사용하거나 linear CRF, beam decoder 를 사용면 된다.



