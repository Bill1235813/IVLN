ob_type=pano
feedback=teacher

features=vitbase_r2rfte2e
# features=vitbase
ft_dim=768

ngpus=1
seed=0

resume=../saved_models/R2R/vitbase-finetune-freeze-embeds-il-mmmtvln/ckpts/best_val_unseen
outdir=../saved_models/R2R/vitbase-finetune-freeze-embeds-il-mmmtvln #-testing

flag="--root_dir ../datasets
      --output_dir ${outdir}

      --dataset r2r
      --mmvl
      --latent_dim 128

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}

      --world_size ${ngpus}
      --seed ${seed}

      --num_l_layers 9
      --num_x_layers 4

      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding
      --fix_obs_embedding
      --fix_mm_embedding

      --features ${features}
      --feedback ${feedback}
      

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 100000
      --log_every 2000
      --batch_size 128
      --optim adamW

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5

      "

      # --mmvl 
      # --latent_dim 128
      # out dir
      # --resume_file ${resume} --resume_optimizer
      # --fast_epoch
      # --iters 100000
      # --log_every 2000
      # --batch_size 2

export PYTHONPATH=../:$PYTHONPATH
## train #0000 000
## vitbase.e2e bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt
# --eval_first
CUDA_VISIBLE_DEVICES=0 python -m pdb r2r/main.py $flag  \
     --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
     --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt

# inference
# vitbase.e2e resume_file: ../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen
# CUDA_VISIBLE_DEVICES=3 python r2r/main.py $flag \
#       --resume_file ../saved_models/R2R/vitbase-finetune-freeze-embeds-il-mmmtvln/ckpts/best_val_unseen \
#       --test --submit
