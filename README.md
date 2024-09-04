# Bubble-Cavitation-ML

Repository for the paper "Classifying Acoustic Cavitation with Machine Learning Trained on Multiple Physical Models." This includes datasets, models, and analysis scripts.

## Abstract

Acoustic cavitation refers to the formation and oscillation of microbubbles in a liquid exposed to acoustic waves. Depending on the properties of the liquid and the parameters of the acoustic waves, bubbles behave differently. The two main regimes of bubble dynamics are transient cavitation, where a bubble collapses violently, and stable cavitation, where a bubble undergoes periodic oscillations. Predicting these regimes under specific sonication conditions is important in biomedical ultrasound and sonochemistry. For these predictions to be helpful in practical settings, they must be precise and computationally efficient. In this study, we have used machine learning techniques to predict the cavitation regimes of air bubble nuclei in a liquid. The supervised machine learning was trained by solving three differential equations for bubble dynamics, namely the Rayleigh-Plesset, Keller-Miksis, and Gilmore equations. These equations were solved for a range of initial parameters, including temperature, bubble radius, acoustic pressure, and frequency. Four different classifiers were developed to label each simulation as either stable or transient cavitation. Subsequently, four different machine-learning strategies were designed to analyze the likelihood of transient or stable cavitation for a given set of acoustic and material parameters. Cross-validation on held-out test data shows a high accuracy of the machine learning predictions. The results indicate that machine learning models trained on physics-based simulations can reliably predict cavitation behavior across a wide range of conditions relevant to real-world applications. This approach can be employed to optimize device settings and protocols used in imaging, therapeutic ultrasound, and sonochemistry.

## Installation

The library can be used from the Jupyter notebooks in the `examples` folder with a recent Python installation and common packages. The necessary Python packages can be installed via `pip` or `conda`.

```
conda install joblib jupyter matplotlib numpy pandas scikit-learn scipy seaborn tabulate
```

## Citation

If you use this repository in your research, please cite the following:

### BibTeX

```bibtex
@misc{gatica2024cavitation,
  title={Classifying Acoustic Cavitation with Machine Learning Trained on Multiple Physical Models},
  author={Gatica, Trinidad and van 't Wout, Elwin and Haqshenas, Reza},
  year={2024},
  note={Preprint available on arXiv:2408.16142},
  doi={10.48550/arXiv.2408.16142},
  institution={{Pontificia Universidad Católica de Chile, Santiago, Chile and University College London, London, United Kingdom}}
}
````

### APA 

Gatica, T., van 't Wout, E., & Haqshenas, R. (2024). "Classifying acoustic cavitation with machine learning trained on multiple physical models." Preprint available on arXiv:2408.16142. DOI: [10.48550/arXiv.2408.16142](https://doi.org/10.48550/arXiv.2408.16142)
