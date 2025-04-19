## Adversarial attack examples on machine learning-aided visualizations

About
-----

* Source code used to perform adversarial attack examples in **[Adversarial Attacks on Machine Learning-Aided Visualizations](https://doi.org/10.1007/s12650-024-01029-2)**, Takanori Fujiwara, Kostiantyn Kucher, Junpeng Wang, Rafael M. Martins, Andreas Kerren, and Anders Ynnerman.
Journal of Visualization, vol. 28, pp. 133–151, 2025.

******

Content
-----
* attacks_umap.py: Attacks on parametric UMAP (Sec. 4.1).
* attaks_multivision.py: Attacks on MultiVision (Sec. 4.2).
* attacks_umap_multivision.py: Attacks on parametric UMAP x MultiVision (Sec. 4.3).
* attacks_tsne.py: Attacks on parametric t-SNE (the supplementary materials).

* docs/supp_material.pdf: Supplementary materials.

******

Setup
-----

### Requirements
* Python3 (< 3.13: tensorflow does not support 3.13 yet)
* Note: Tested on macOS Sequoia.

### Setup

* Install required python packages: `python3 -m pip install -r requirements.txt`
    
* For attacks on parametric t-SNE:

    - Download the Parametric-DR repository https://github.com/a07458666/parametric_dr and copy `parametric_dr` directory that includes `tsne_nn.py` ti this `attack-ml4vis` directory.
      - `tsne_nn.py` implementation (as of April 19, 2025) doesn't work well with currently available pytorch versions.  To address this issue, 
      
        replace `idx = idxs[i:i+batch_size]` (line 87) 
        
        with `idx = idxs[i:i+batch_size].numpy()` 


### How to run

* After finishing the setup, move to the directory containing attacks_*.py and run either of them.

  - Note: As parametric UMAP and parametric t-SNE involve randomness during the training, generated results may have some differences from those presented in the paper.

******
How to cite
-----

Takanori Fujiwara, Kostiantyn Kucher, Junpeng Wang, Rafael M. Martins, Andreas Kerren, and Anders Ynnerman, "Adversarial Attacks on Machine Learning-Aided Visualization." Journal of Visualization, vol. 28, pp. 133–151, 2025.
