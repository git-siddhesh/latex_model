local rank


If your training script reads local rank from a --local-rank cmd argument. Change your training script to read from the LOCAL_RANK environment variable as demonstrated by the following code snippet:

#### torch.distributed.launch
```
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", type=int)
args = parser.parse_args()
```

#### torchrun

```
local_rank = args.local_rank
import os
local_rank = int(os.environ["LOCAL_RANK"])
```

Usage- 

```
local_rank = int(os.environ["LOCAL_RANK"])
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[local_rank],
                                                  output_device=local_rank)
```

# Gradient Checkpointing 

[Medium BLOG](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)
[GITHUB REPO](https://github.com/cybertronai/gradient-checkpointing)

## 1. Vanilla backprop
simple feed-forward neural network with n layers
![](https://github.com/cybertronai/gradient-checkpointing/raw/master/img/backprop.png)


the order in which these nodes are computed. The purple shaded circles indicate which of the nodes need to be held in memory at any given time.
![](https://github.com/cybertronai/gradient-checkpointing/raw/master/img/output.gif)

## 2; Memory Poor backprop

if we are willing to recompute nodes we can potentially save a lot of memory. We might for instance simply recompute every node from the forward pass each time we need it
![](https://github.com/cybertronai/gradient-checkpointing/blob/master/img/output_poor.gif)

Using this strategy, the memory required to compute gradients in our graph is constant in the number of neural network layers n, which is optimal in terms of memory. However, note that the number of node evaluations now scales with n^2, whereas it previously scaled as n: Each of the n nodes is recomputed on the order of n times. 

## 3; Checkpointing
To strike a balance between memory and computation

![](https://github.com/cybertronai/gradient-checkpointing/raw/master/img/checkpoint.png)

These checkpoint nodes are kept in memory after the forward pass, while the remaining nodes are recomputed at most once  
After being recomputed, the non-checkpoint nodes are kept in memory until they are no longer required  

For the case of a simple feed-forward neural net, all neuron activation nodes are graph separators or articulation points of the graph defined by the forward pass. This means that we only need to recompute the nodes between a b node and the last checkpoint preceding it when computing that b node during backprop. When backprop has progressed far enough to reach the checkpoint node, all nodes that were recomputed from it can be erased from memory. The resulting order of computation and memory usage then look as follows

![](https://github.com/cybertronai/gradient-checkpointing/raw/master/img/output2.gif)