# Weakly Supervised EMCAD

This repo is based on [EMCAD](https://github.com/SLDGroup/EMCAD/).


## Architecture

<p align="center">
<img src="model_architecture.png" width=100% height=40% 
class="center">
</p>


## Qualitative Results

<p align="center">
<img src="qualitative_results_clinicdb.png" width=80% height=25% 
class="center">
</p>

## Usage:
### Recommended environment:
**Please run the following commands.**
```
conda create -n emcadenv python=3.8
conda activate emcadenv

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

pip install -r requirements.txt

```

### Data preparation:
- **Synapse Multi-organ dataset:**
Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and download the dataset. Then split the 'RawData' folder into 'TrainSet' (18 scans) and 'TestSet' (12 scans) following the [TransUNet's](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) lists and put in the './data/synapse/Abdomen/RawData/' folder. Finally, preprocess using ```python ./utils/preprocess_synapse_data.py``` or download the [preprocessed data](https://drive.google.com/file/d/1wvmw8DVyDKr5sOAFn5zUpfhbK4Vxjze4/view) and save in the './data/synapse/' folder. 
Note: If you use the preprocessed data from [TransUNet](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd), please make necessary changes (i.e., remove the code segment (line# 88-94) to convert groundtruth labels from 14 to 9 classes) in the utils/dataset_synapse.py. 

- **ACDC dataset:**
Download the preprocessed ACDC dataset from [Google Drive](https://drive.google.com/file/d/1CruCQ-jjvA97BX-LIYwXaRMLmp3DN9zc/view) and move into './data/ACDC/' folder.

- **Polyp datasets:**
Download the splited polyp datasets from [Google Drive](https://drive.google.com/drive/folders/1XyjNgmPqikGxCaOdP0i6Xzf3deDIpbCV?usp=share_link) and move into './data/polyp/' folder.

### Pretrained model:
You should download the pretrained PVTv2 model from [Google Drive](https://drive.google.com/drive/folders/1d5F1VjEF1AtTkNO93JwVBBSivE8zImiF?usp=share) or [PVT GitHub](https://github.com/whai362/PVT/releases/tag/v2), and then put it in the './pretrained_pth/pvt/' folder for initialization.

## Weakly Supervised Learning Setup

This repository supports weakly supervised training for medical image segmentation, aiming to reduce manual labeling costs. Two modes of weak supervision are implemented:


####  Scribble Supervision
* **Description**: Uses a skeletonization algorithm (`skimage.morphology.skeletonize`) to thin the ground-truth binary mask down to a **single-pixel-wide backbone line** (a continuous scribble). For background supervision, it randomly samples up to 500 points.
* **Effect**: Simulates actual human clinician scribbles, requiring the model to generalize boundaries without explicit edge annotations during training.

---

### 2. Loss Function Modification (`structure_loss_weak`)

To train with weak annotations, we use `structure_loss_weak` which masks out the unlabeled regions. Pixels that are not annotated are assigned an ignore label of **`255`**.

The loss is calculated only on the annotated pixels (`valid` mask):

---

### Training:
To train the model on the Polyp dataset using weak supervision, run:
```bash
python train_polyp.py \
    --encoder pvt_v2_b2 \
    --pretrained_dir ./pretrained_pth/pvt/ \
    --train_path ./data/polyp/target/ClinicDB/train/ \
    --test_path ./data/polyp/target/ClinicDB/ \
    --epoch 200 \
    --batchsize 8
```

### Trained Weights on Synapse Dataset:
You can download the trained weights on Synapse dataset from [Google Drive](https://drive.google.com/drive/folders/1S-hxcgMlTFEX9GJGTUF7XWdZBGx7MiZl?usp=sharing).   

### Testing:
To evaluate the trained polyp segmentation model, run:
```bash
python test_polyp.py --encoder pvt_v2_b2
```

## Acknowledgement
We are very grateful for these excellent works [timm](https://github.com/huggingface/pytorch-image-models), [CASCADE](https://github.com/SLDGroup/CASCADE), [MERIT](https://github.com/SLDGroup/MERIT), [G-CASCADE](https://github.com/SLDGroup/G-CASCADE), [PP-SAM](https://github.com/SLDGroup/PP-SAM), [PraNet](https://github.com/DengPingFan/PraNet), [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) and [TransUNet](https://github.com/Beckschen/TransUNet), which have provided the basis for our framework.

## Citations

``` 
@inproceedings{rahman2024emcad,
  title={Emcad: Efficient multi-scale convolutional attention decoding for medical image segmentation},
  author={Rahman, Md Mostafijur and Munir, Mustafa and Marculescu, Radu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11769--11779},
  year={2024}
}
```
