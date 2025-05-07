# Weight Refiner

This repository contains the implementation of the method presented in the paper [Stay Positive: Neural Refinement of Sample Weights](https://arxiv.org/abs/2505.03724).
The approach introduces a novel method for transforming sample weights in a weighted dataset, with the goal of rendering them all positive.

## 📄 Paper

The method is detailed in the paper:

> **Stay Positive: Neural Refinement of Sample Weights** [arXiv\:2505.03724](https://arxiv.org/abs/2505.03724)

## 🧪 Usage

The repository includes several Jupyter notebooks demonstrating the application of the weight refinement technique:

* `nb_gauss_easy.ipynb`: Simple Gaussian distribution example.
* `nb_gauss_negative.ipynb`: Gaussian distribution with negative weights.
* `nb_gauss_spread.ipynb`: Gaussian distribution with spread weights.
* `nb_tt_ensemble.ipynb`: Ensemble method with weight refinement.
* `nb_weight_shape.ipynb`: Analysis of weight shapes.
* `schema.ipynb`: Schema and data preprocessing.([GitHub Docs][1])

These notebooks provide practical examples and visualizations to understand the impact of refined sample weights.

## 🔧 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Nollde/weight_refiner/blob/main/LICENSE) file for more details.
