# Weight Refiner

This repository contains the implementation of the neural weight refinement method presented in the paper **Stay Positive: Neural Refinement of Sample Weights** ([arXiv\:2505.03724](https://arxiv.org/abs/2505.03724)).
The approach introduces a novel method for transforming sample weights in a weighted dataset, with the goal of rendering them all positive.

## 📄 Paper

**Stay Positive: Neural Refinement of Sample Weights** ([arXiv\:2505.03724](https://arxiv.org/abs/2505.03724))

>Monte Carlo simulations are an essential tool for data analysis in particle physics. Simulated events are typically produced alongside weights that redistribute the cross section across the phase space. Latent degrees of freedom introduce a distribution of weights at a given point in the phase space, which can include negative values. Several post-hoc reweighting methods have been developed to eliminate the negative weights. All of these methods share the common strategy of approximating the average weight as a function of phase space. We introduce an alternative approach with a potentially simpler learning task. Instead of reweighting to the average, we refine the initial weights with a scaling transformation, utilizing a phase space-dependent factor. Since this new refinement method does not need to model the full weight distribution, it can be more accurate. High-dimensional and unbinned phase space is processed using neural networks for the refinement. Using both realistic and synthetic examples, we show that the new neural refinement method is able to match or exceed the accuracy of similar weight transformations.

## 🧪 Usage

The codebase of the neural weight refinement can be found in the directory [refiner](https://github.com/Nollde/weight_refiner/tree/main/refiner).
It includes the neural weight refinement technique from the paper and the related method of neural reweighting ([2007.11586](https://arxiv.org/abs/2007.11586)).

Additionally, the repository includes several Jupyter notebooks demonstrating the application of the weight refinement technique.
Each notebook presents one case study in the paper

* A) Top Quark Pair Production: `nb_tt_ensemble.ipynb`
* B) Synthetic Shape Example: `nb_weight_shape.ipynb`
* C) Synthetic Spectrum Example: `nb_gauss_spread.ipynb`
* D) Synthetic Extrapolation Example: `nb_gauss_easy.ipynb`
* E) Synthetic Negative Density Example: `nb_gauss_negative.ipynb`

## 🔧 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Nollde/weight_refiner/blob/main/LICENSE) file for more details.
