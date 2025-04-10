# Uncertainty Quantification In Machine Learning Based Retrieval Of Soil Moisture From GNSS-R Observations

While microwave imaging satellites, such as the NASA Soil Moisture Active Passive (SMAP), can provide reliable estimates
of surface soil moisture at km resolution, the temporal frequency of observations is on the order of days.
To increase the temporal frequency of observations, a new class of approaches considers global navigation satellite system
(GNSS)-reflectometry (GNSS-R) signals. In this work, we consider observations from the NASA Cyclone GNSS
(CYGNSS) constellation, as well as auxiliary observations, and seek to provide instantaneous soil moisture estimates.
To achieve accurate retrievals, a novel machine learning approach for probabilistic regression is considered, namely the
NGBoost. In addition to achieving an accuracy comparable to previous approaches employing state-of-the-art machine
learning methods, the considered framework also provides prediction intervals to quantify prediction uncertainty. Using
observations from the Yanco SMAP core validation site in southeast Australia over a period of three years, we quantify
the performance in terms of both retrieval accuracy and associated uncertainty. Furthermore, using noisy observations,
we experimentally demonstrate the impact of input noise on the prediction uncertainty.

## Citation

If you use this code or find our work useful in your research, please consider citing our paper:

```bibtex
@article{tsagkatakis2024uncertainty,
  author = {G. Tsagkatakis et al.},
  title = {Uncertainty Quantification in Machine Learning Based Retrieval of Soil Moisture From GNSS-R Observations},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year = {2024},
  volume = {62},
  doi = {10.1109/TGRS.2024.3387452},
  url = {https://ieeexplore.ieee.org/abstract/document/10642241}
}
