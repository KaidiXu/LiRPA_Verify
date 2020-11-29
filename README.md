# Fast and Complete: Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers

<p align="center">
<img src="http://huan-zhang.com/images/paper/lirpa_verify.png" width="100%" height="100%">
</p>


We propose to use linear relaxation based perturbation analysis ([LiRPA](https://arxiv.org/pdf/2002.12920)) to replace Linear Programming (LP) during the branch and bound (BaB) process in complete neural network verification. 
LiRPA based algorithms (e.g.  [CROWN](https://arxiv.org/pdf/1811.00866.pdf) and
[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf)) can be efficiently implemented on typical machine learning accelerators such as GPUs and TPUs, and we exploit the automatic, highly efficient and differentiable implementation [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA/). 
We apply a fast gradient based bound tightening procedure combined with batch splits and the design of minimal usage of LP, which enable us to effectively use LiRPA on the accelerator hardware for the challenging complete NN verification problem. We significantly outperform LP-based approaches: on a single GPU, we demonstrate **an order of magnitude speedup** compared to existing LP-based approaches.

Please refer to our paper for more details:

Kaidi Xu*, Huan Zhang*, Shiqi Wang, Yihan Wang, Suman Jana, Xue Lin and Cho-Jui Hsieh, "_Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers_", [pdf](https://huan-zhang.com/pdf/RapidVerification.pdf) (\* Equal contribution)

Our implementation is based on the [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA/) library. We are working on finalizing our code and will release full source code soon.
