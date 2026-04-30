<img src="docs/images/readme.png" alt="Multimodal AI for Alzheimer's Disease"/>

<p align="center">
    <strong>A Quantitative Analysis of Multimodal Biomarkers in Alzheimer’s Disease</strong><br/>
    <a href="https://scholar.google.com/citations?user=VNQ6auUAAAAJ">Antonio Scardace</a> and
    <a href="https://scholar.google.com/citations?user=qh2Rr-cAAAAJ">Daniele Ravì</a>
</p>

<div align="center">
    <a href="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge" alt="Python"><img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge" alt="Python"></a>
    <a href="https://ieeexplore.ieee.org/document/11740084"><img src="https://img.shields.io/badge/Paper-pdf-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&color=94DD15" alt="Paper PDF"></a>
    <a href="https://github.com/antonioscardace/Multimodal-AD/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" alt="License"></a>
</div>
<br/>

Modern Alzheimer's disease (AD) research increasingly relies on data from multiple biological domains, yet how these modalities **overlap**, what information each **uniquely contributes**, and which are **most useful for predicting others** remain poorly understood despite the high cost and burden these measurements place on patients.

In our paper, we analyze a multimodal cohort comprising **tau-PET**, **T1w sMRI**, **MMSE**, **CDR**, and **APOE $\varepsilon4$** status, and provide a reproducible pipeline that can be reused on richer multimodal datasets. The pipeline quantifies cross‑modal information sharing, investigates predictive relationships between modalities, characterizes structure–pathology associations, and models the main trajectory of disease progression using [**SuStAIn**](https://github.com/ucl-pond/pySuStaIn). The multimodal dataset has been preprocessed using the [**neuroflux**](https://github.com/antonioscardace/neuroflux) tool.

## Citation

Please, reference this publication if you find this code useful:

```BibTeX
@inproceedings{scardace2026quantitative,
    author={Scardace, Antonio and Ravì, Daniele},
    booktitle={2026 IEEE Symposium on Computers and Communications (ISCC)}, 
    title={A Quantitative Analysis of Multimodal Biomarkers in Alzheimer’s Disease}, 
    year={2026},
    pages={1-6},
    doi={10.1109/ISCC70364.2026.11740084}
}
```