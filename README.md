# Functional Regularization for Representation Learning: A Unified Theoretical Perspective

This is the official repository accompanying the NeurIPS 2020 Paper [Functional Regularization for Representation Learning: A Unified Theoretical Perspective](https://arxiv.org/abs/2008.02447). 

![Representation Learning via Functional Regularization](/images/func_reg.png)

## Auto-Encoder

This directory contains:
- The scripts to generate synthetic data with the properties described in Section 5.1 of the paper
- The scripts to run end-to-end training and training with functional regularization via an auto-encoder
- The scripts to plot the t-SNE visualization graphs for the functional approximations

## Masked Self-Supervision

This directory contains:
- The scripts to generate synthetic data with the properties described in Section 5.2 of the paper
- The scripts to run end-to-end training and training with functional regularization via masking the first input component
- The scripts to plot the t-SNE visualization graphs for the functional approximations

## Additional Experiments: Fashion-MNIST

Work-in-progress.

## Additional Experiments: MRPC

Use the [transformers](https://github.com/huggingface/transformers) code repository released by HuggingFace.

The MRPC dataset can be accessed [here](https://www.microsoft.com/en-us/download/details.aspx?id=52398).

## Citation 

If you find this code or our paper useful, please consider citing us using: 

```
@article{garg20functional,
  author       = {Siddhant Garg and Yingyu Liang},
  title        = {Functional Regularization for Representation Learning: A Unified Theoretical Perspective},
  conference   = {NeurIPS 2020},
  url          = {https://arxiv.org/abs/2008.02447},
}
```

## Contact

For direct communication, please contact me (sidgarg is at amazon dot com) if you have any questions regarding the code or the experiments.
