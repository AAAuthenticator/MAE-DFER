server=164
pretrain_dataset='voxceleb2'
pretrain_server=170
finetune_dataset='CAS3_MA'
num_labels=2
ckpts=(99)
input_size=160
sr=1
model_dir="videomae_pretrain_base_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_server${pretrain_server}"

splits=(1)
for ckpt in "${ckpts[@]}";
do
  for split in "${splits[@]}";
  do
    OUTPUT_DIR="./saved/model/finetuning/${finetune_dataset}/${pretrain_dataset}_${model_dir}/checkpoint-${ckpt}/eval_split0${split}_lr_1e-3_epoch_100_size${input_size}_sr${sr}_server${server}"
    if [ ! -d "$OUTPUT_DIR" ]; then
      mkdir -p $OUTPUT_DIR
    fi
    # path to split files (train.csv/val.csv/test.csv)
    DATA_PATH="./saved/data/${finetune_dataset}"
    #"./saved/data/${finetune_dataset}/org/split0${split}"
    # path to pre-trained model
    MODEL_PATH="./saved/model/pretraining/${pretrain_dataset}/${model_dir}/checkpoint-${ckpt}.pth"

    # batch_size can be adjusted according to number of GPUs
    # this script is for 2 GPUs (1 nodes x 4 GPUs)
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 \
        --master_port 13251 \
        run_class_finetuning.py \
        --model vit_base_patch16_${input_size} \
        --data_set ${finetune_dataset^^} \
        --nb_classes ${num_labels} \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 10 \
        --num_sample 1 \
        --input_size ${input_size} \
        --short_side_size ${input_size} \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate ${sr} \
        --opt adamw \
        --lr 1e-3 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 20 \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 2 \
        #--num_workers 16 \
#        >${OUTPUT_DIR}/nohup_rerun.out 2>&1
  done
done
echo "Done!"

