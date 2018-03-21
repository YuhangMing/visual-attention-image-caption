# visual-attention-image-caption

## References
This pytorch implementation is based on Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." International Conference on Machine Learning. 2015. available at https://arxiv.org/pdf/1502.03044.pdf
Author's theano code: https://github.com/kelvinxu/arctic-captions 

The dataset I used is the Flickr8K dataset.

## Results
 
<br/>

#### Stochastic Hard Attention Model

##### (1) Generated caption: a skier go across the snow .
![alt text](results/hard_1.png "hard example 1")

##### (2) Generated caption: a woman in the goggles .
![alt text](results/hard_2.png "hard example 2")

##### BLEU Scores:
![alt text](results/hard-box.png "box plot of the BLEU score")

#### Deterministic Soft Attention Model

##### (1) Generated caption: a dog jump over a stick .
![alt text](results/soft_1.png "soft example 1")

##### (2) Generated caption: a man ride his moterocycle down a sidewalk .
![alt text](results/soft_2.png "soft example 2")

##### BLEU Scores:
![alt text](results/soft-box.png "box plot of the BLEU score")
