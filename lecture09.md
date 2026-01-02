# Lecture 5: Scaling Laws 1

Scaling laws allow us to roughly estimate the loss that we can expect from a given model size, compute, and data.

A toy example:
If we take an average, the 'scaling law' can be computed as:
log(Error) = -log n + 2 log mu

In Deep learning, it's not always as simple.


## Scaling laws exist for many factors

On a log-log scale we observe linear relationships:
Compute - Test Loss
Dataset Size - Test Loss
Parameters - Test Loss

This even holds if train and test data is different. Like downstream tasks, etc.

## Data vs Performance

Loss and Dataset size is linear on a log-log plot (=powerlaw). --> More data is better.
This being a powerlaw means that the error naturally decays polynomially.

In practice we have finite data, reusing the data in multiple epochs gives diminishing returns, there are also scaling laws for this.
--> Should we repeat good data, or add more data but its lower quality. Generally: smaller compute means highly aggressive filtering is best; big compute, less filtering is best.

## Kaplan's observations when looking at scaling laws

Q: Are transformers better than LSTMs?
A: Kaplan showed that even with rising layers and parameters, transformers are always better.
Tay et al even looked at much more architectures, only switch transformers (i.e. MoE) is better.

Q: Is SGD or ADAM better
A: Kaplan showed with dataset size, ADAM is always better

Q: What is better, Deep or Wide models?
A: Layer choice (Aspect Ratio) seems to be not as clear. There is a wide basin where you are kind of optimal.
--> But changing embedding parameters size behaves differently, here more is better.

In general, this knowledge is important to make architecture choices to predict what is going to help.

Batch size: larger batch size does help, but only to a certain point, after which it has diminishing returns.
--> Critical batch size, is basically right before the diminishing returns start.
--> The smaller the loss target, the bigger the batch size should be. Some models change batch size after a while training.

Learning rate: The bigger the model (how wide it is), the smaller the optimal learning rate should be.
muP: method that adjusts the LR of different model sections, and this allows to first choose the LR on a small model, which will then in automatically scale to a bigger model.

## Downstream scaling

Perplexity/Loss on a text dataset is nicely log-log linearly correlated, but downstream eval's are more scattered.

## Scaling law based design:

1. Train a few smaller models
2. Establish a scaling law (e.g. ADAM vs SGD scaling law)
3. Select optimal hyperparam based on the scaling law prediction

## Chinchilla

Chinchilla (from google) says that these scaling laws are off, and said that the right tradeoff between tokens and parameters are different.

Chinchilla aims to tell you what gives the best model for fixed training compute. However, if the model is actually used, most of the compute is gonna be used in inference. So we should overtrain! The 20 tokens/param is therefore quite low. Llama 3 has 215 tokens/param. --> Having a better smaller model (inference takes less compute) is better than having a bigger model that used less compute for training (but more for inference).