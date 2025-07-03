# [SuPa-Match: High-Resolution 3D Shape Matching with Global Optimality and Geometric Consistency](https://diglib.eg.org/handle/10.1111/cgf70208)

Official repository for the SGP 2025 paper: High-Resolution 3D Shape Matching with Global Optimality and Geometric Consistency by Nafie El Amrani, Paul Roetzer and Florian Bernard (University of Bonn & Lamarr Institute).

 We compute geometrically consistent matchings between two (high-resolution) 3D shapes with our two-stage pipeline:
1. We solve our optimisation problem **PATCH-MATCH** to find a geometrically consistent matching of (automatically computed) patches on the source shape and the target shape (patches are *only* defined on the source shape)
2. For every corresponding patch we solve our optimisation problem **SURF-MATCH** to find a dense and geometrically consistent matching between both patches while respecting edge-correspondences from stage 1  

## ⚙️ Installation

### Prerequisites 
You need a working c++ compiler and cmake. Note: builds are only tested on unix machines.

### Installation Step-by-Step
1. Create python environment
```shell
conda create -n supamatch python=3.8
conda activate supamatch
pip3 install torch torchvision torchaudio
git clone git@github.com:NafieAmrani/SuPa-Match.git
cd SuPa-Match
pip install -r requirements.txt
```

2. Install GeCo (code to create the Patch-Match & Surf-Patch integer linear programs)
```shell
git clone git@github.com:paul0noah/GeCo.git
cd GeCo
python setup.py install
cd ..
```

3. Retrieve a gurobi license from the [official website](https://www.gurobi.com/).

## 📝 Dataset
Datasets are available from this [link](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view). Put all datasets under `./datasets/` such that the directory looks somehow like this Two example files for `FAUST_r` shapes are included in this repository.

```
├── datasets
    ├── FAUST_r
    ├── SMAL_r
    ├── DT4D_r
```
We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the original authors.

## 🧑‍💻️‍ Usage
See `supamatch_example.py` for example usage.

## 🙏 Acknowledgement
The implementation of DiffusionNet is based on the [official implementation](https://github.com/nmwsharp/diffusion-net). The framework implementation is adapted from [Unsupervised Deep Multi Shape Matching](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching). This repository is adapted from [Unsupervised-Learning-of-Robust-Spectral-Shape-Matching](https://github.com/dongliangcao/Unsupervised-Learning-of-Robust-Spectral-Shape-Matching) and [SpiderMatch: 3D Shape Matching with Global Optimality and Geometric Consistency](https://github.com/paul0noah/spider-match) .

## 🎓 Attribution
```
@article{elamrani2025highres,
    journal = {Computer Graphics Forum},
    title = {{High-Resolution 3D Shape Matching with Global Optimality and Geometric Consistency}},
    author = {El Amrani, Nafie and Roetzer, Paul and Bernard, Florian},
    year = {2025},
    publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
    ISSN = {1467-8659},
    DOI = {10.1111/cgf.70208}
}
```

## 🚀 License
This repo is licensed under MIT licence.
