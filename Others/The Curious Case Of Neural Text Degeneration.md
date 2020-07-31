# THE CURIOUS CASE OF NEURAL TENXT DEGENERATION



## 1. Introduction

* gpt2는 *tok-k* sampling을 사용한 모델이다. 
* beam-search와 같은 high probability 기반 decode 방법은 gpt-2와 높은 성능의 모델에서도 generic(포괄적), repetitive and awkward(어색한)한 텍스트를 output하게 할 수 있다.
* sampling 방법은 incoherent하고 context과 관련없는 문장을 만들 수 있다. `unreliable tail`이 문장을 구성하면서 candidate token with relatively low probability가