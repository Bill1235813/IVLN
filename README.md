# Iterative VLN for R2R
(copied from Jacob)

The rough new evaluation paradigm is as follows:

For M iterations:
1. Select a random permutation of trajectories in the scene.
1. Teleport to starting position A0 of trajectory 0. Set i=0.
1. Performance inference from node Ai to node Bi with language instruction. Update evaluation metrics and walk from Bi* (inferred stop point) to Bi with no language instruction.
1. Walk from Bi to B(i+1) with no language instruction [image the robot is following a person].
1. Set i:=i+1 and return to (3).


## Installation

1. Install requirements:
```setup
conda create --name vlnhamt python=3.8.5
conda activate vlnhamt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
2. Download R2R data from [Dropbox](https://www.dropbox.com/sh/3a5j03u286px604/AABNp887W7_Fhgv13gUt4wzda?dl=0), including processed annotations, features and pretrained models. Put the data under *datasets* directory.

3. Download Matterport 3D adjacency maps and angle features from [Dropbox](https://www.dropbox.com/sh/1jibefgj956rjbp/AAAx-ATXwrPk6NlLKFUW6DFsa?dl=0). Put the files under *datasets* directory. 
4. Download the tour files for the original VLN-R2R and the prevalent augmented data from [GDrive](https://drive.google.com/drive/folders/1pALNPuAdSxtAKpUel9BNuy0Dn11_PZNP?usp=sharing). Put the files under the *iterative-vln* directory. The directory structure after this will be like
```directory
iterative-vln
|___finetune_src
|___datasets
|   |___R2R
|   |___total_adj_list.json
|   |___angle_feature.npy
|   |___...
|___tours_iVLN.json
|___tours_iVLN_prevalent.json
|___...
```


## Run the baseline
HAMT Teacher-forcing IL
```baseline
cd finetune_src
bash scripts/run_r2r_il.sh
```


## Run the extended memory experiments
```ext_mem
cd finetune_src
bash scripts/iter_train.sh
bash scripts/iter_train_sep.sh              # with prev hist identifier
bash scripts/iter_train_hist.sh             # train hist encoder
bash scripts/iter_train_sep_hist.sh         # with prev hist identifier and train hist encoder
bash scripts/iter_train_sep_weight.sh       # with prev hist identifier and inflection weighting
bash scripts/iter_train_sep_hist_weight.sh  # all three above
```


## Citation
If you find this work useful, please consider citing:


## Acknowledgement
Some of the codes are built upon [HAMT](https://github.com/cshizhe/VLN-HAMT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [UNITER](https://github.com/ChenRocks/UNITER) and [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT).
Thanks them for their great works!
