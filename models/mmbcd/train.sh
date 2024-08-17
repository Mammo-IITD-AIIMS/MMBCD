CUDA_VISIBLE_DEVICES=4,5 nohup python ./code/train.py \
    --checkpoint_model_save './models/mmbcd/' \
    --topk 8 \
    --num_epochs 128 \
    --num_workers 16 \
    --batch_size 16 \
    --rob_layers_unfreeze 2 \
    --vit_layers_freeze 9 \
    --img_size 224 > ./models/mmbcd/train_logs.txt &