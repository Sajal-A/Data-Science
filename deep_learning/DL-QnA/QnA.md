# Computational Considerations
**Feed-Forward Networks:**
- Simple Strucuture: Feed-Forward networks follow a straight path from input to output. This makes them easier to implement and tune.
- Parallel Computation: Inputs can be processed in batches, enabling fast training using modern hardware.
- Efficient Backpropagation: They use standard backpropagation which is stable and well-supported across frameworks.
- Lower Resource Use: No memory of past inputs means less overhead during training and inference.

  **Recurrent Neural Networks**
  - Sequential Natute: RNNs process data step-by-step, this limits parallelism and slows down training.
  - Harder to Train: Training uses Backpropagation Through Time (BPTT) which can be unstable and slower.
  - Captures Temporal Patterns: They are suited for sequential data but require careful tuning to learn long-term dependencies.
  - Higher Compute Demand: Maintaining hidden states and learning overt time steps makes RNNs more resource-intesive.
 
**Primary Difference b/w feed forward neural network and RNN**
- Inputs and outputs do not have a fixed length, i.e., some input sentences can be of 10 words while others could be not as same. The same is true for the eventual output.
- We will not be able to share feature learned across different positions of text if we use a standard neural network.
