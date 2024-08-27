# Bubble-Cavitation-ML

Repository for the paper "Classifying Bubble Cavitation with Machine Learning Trained on Multiple Physical Models." This includes datasets, models, and analysis scripts.

## Abstract

Predicting the type of cavitation that a bubble will undergo under specific conditions is crucial for several problems in biomedical and sonochemical applications. An important example is distinguishing between transient and stable cavitation. These predictions of cavitation behavior must be accurate and performed with reasonable computational effort to be relevant in practice.

In this study, we apply algorithms based on machine learning to predict the cavitation type of air bubbles in water. We train the supervised machine learning models by solving three differential equations for bubble dynamics, namely the Rayleigh-Plesset, Keller-Miksis, and Gilmore equations. These cavitation models are numerically solved for a range of initial parameters, including the bubble radius, acoustic pressure, wave frequency, and temperature.

We develop four different classifiers to label each simulation as either stable or transient cavitation. We then design four different machine-learning strategies to analyze the likelihood of transient or stable cavitation for a given set of acoustical and material parameters. Cross-validation on held-out test data shows a high accuracy of the machine learning predictions.

Hence, this study confirms the feasibility of using machine learning trained on physical models to reliably distinguish between stable and transient cavitation for a wide range of parameters relevant to practical situations. This predictive capability can be employed to optimize settings in devices used in biomedical and sonochemical applications and provides a valuable tool for determining the likelihood of bubbles undergoing either transient or stable cavitation.

## Citation

If you use this repository in your research, please cite the following:

### BibTeX

```bibtex
@article{gatica2024cavitation,
  title={Classifying Bubble Cavitation with Machine Learning Trained on Multiple Physical Models},
  author={Gatica, Trinidad and van 't Wout, Elwin and Haqshenas, Reza},
  journal={Journal of the Acoustical Society of America},
  year={2024},
  note={Submitted for publication},
  institution={{Pontificia Universidad Católica de Chile, Santiago, Chile and University College London, London, United Kingdom}}
}
````

### APA 

Gatica, T., van 't Wout, E., & Haqshenas, R. (2024). Classifying bubble cavitation with machine learning trained on multiple physical models. Journal of the Acoustical Society of America. (Submitted for publication).