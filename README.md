# Higher order DGNet: Discrete gradient methods in Hamiltonian neural networks

This code is meant as a supplement to [1], and is an implementation of the higher order discrete gradient methods for Hamiltonian neural networks presented there. It builds on 
* DGNet by Matsubara et al [3] (https://github.com/tksmatsubara/discrete-autograd), which again builds on
* Hamiltonian neural networks by Greydanus et al. [2] (https://github.com/greydanus/hamiltonian-nn).

Please refer to [1] if the code is used in a project.

[1] S. Eidnes. "Order theory for discrete gradient methods." arXiv preprint, arXiv:2003.08267 (2020).

[2] S. Greydanus, M. Dzamba, and J. Yosinski. "Hamiltonian neural networks." Advances in Neural Information Processing Systems, 32:15379–15389 (2019).

[3] T. Matsubara, A. Ishikawa, and T. Yaguchi. "Deep energy-based modeling of discrete-time physics." arXiv preprint, arXiv:1905.08604 (2019).

## Dependencies
* PyTorch
* NumPy
* Scipy
* Autograd
* Matplotlib
