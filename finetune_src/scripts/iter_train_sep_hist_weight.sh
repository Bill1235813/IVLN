#!/bin/bash -l
#SBATCH --job-name=iterative_il_train_sep    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=wangzhu@usc.edu     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6000:1
#SBATCH --mem=30gb                     # Job memory request
#SBATCH --time=3-0:00:00               # Time limit hrs:min:sec
#SBATCH --output=/home/zhuwangz/iterative-vln/finetune_src/slurm_output/%j.log   # Standard output and error log

ob_type=pano
feedback=teacher

features=vitbase_r2rfte2e
# features=vitbase
ft_dim=768

ngpus=1
seed=0

outdir=../saved_models/R2R/vitbase-finetune-iterative-il-trainhist-sep-weight-b8

flag="--root_dir ../datasets
      --output_dir ${outdir}

      --iterative
      --extended_history
      --sep_hist
      --rebuild

      --dataset r2r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}

      --world_size ${ngpus}
      --seed ${seed}

      --num_l_layers 9
      --num_x_layers 4

      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding

      --features ${features}
      --feedback ${feedback}


      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 10000
      --log_every 200
      --batch_size 8
      --optim adamW

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5

      --inflation_weighting 0.2

      "


export PYTHONPATH=../:$PYTHONPATH
## train #0000 000
## vitbase.e2e bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt
# --eval_first
CUDA_VISIBLE_DEVICES=0 python r2r/main.py $flag  \
     --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
     --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
     --iterative

# inference
# vitbase.e2e resume_file: ../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen
# CUDA_VISIBLE_DEVICES=0 python -m pdb r2r/main.py $flag \
#       --resume_file ../saved_models/R2R/vitbase-finetune-e2e/ckpts/best_val_unseen \
#       --test --iterative
#       --submit
