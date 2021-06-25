
eval_gpt2() {
  python run_clm.py \
    --model_type gpt2 \
    --config_name ./configs/gpt2 \
    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
    --output_dir ./models/${1} \
    --do_train \
    --train_file=${train_file} \
    --do_eval \
    --validation_file=${test_file} \
    --per_device_eval_batch_size ${batch_size}
}

# datasets
train_file=../../datasets/cc/cc_train_6-0_no-symbols.txt
test_file=../../datasets/cc/cc_dev_6-0_no-symbols.txt

# eval
batch_size=4
eval_gpt2 cc-gpt2-BS_4-LR_1e-4-epoch_10
