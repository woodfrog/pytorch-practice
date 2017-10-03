## Basic Functionalities

[Fundamental tutorial](http://pytorch.org/tutorials/beginner/pytorch_with_examples.html)

### Main Features of Pytorch

1. Tensors. It's not like the tensors of Tensorflow, the tensors of pytorch is more like ndarray but it can take advantage of GPUs for acceleration.

2. Automatic differentiation, the same as all modern frameworks

### Autograd

1. In pytorch, tensors are wrapped with Variable. Variable has almost the **same API as tensors** but using it will **build a computing graph in-the-fly during forward pass**, and the gradients in the graph can be automatically computed. 

2. In the pytorch computing graph, nodes are tensors and edges are operations which take input tensors to produce output tensors. (Different from the definition of tensorflow graph, in which tensors and ops are both nodes and edges are tensors passing between them. But it's not that important)

3. Variable x: x.data --> a Tensor contains actual data. x.grad is another Variable holding the gradient of x wrt. some scalar value (always be the loss).

### Dynamic Graph vs. Static Graph

Pytorch's feature of dynamic graph definitely make it easier to implement some dynamic models (which are difficult to implement with tensorflow). For examples RNNs with different unrolling steps for different data.


### nn and optim models

check the tutorial for more details

