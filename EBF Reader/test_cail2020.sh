date
bert_dir='../roberta-base-chinese' # https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main

python run_cail.py \
    --name train_v1 \
    --bert_model $bert_dir \
    --data_dir data \
    --val_dir data_2020 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --lr 3e-5 \
    --seed 1 \
    --k_v_dim 256 \
    --att_back_lambda 3.0 \
    --gradient_accumulation_steps 1 \
    --pre_each_epc 5 \
    --epc_start_pre 7 \
    --epochs 10 \

date