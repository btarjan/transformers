
## CC GPT-2 finetuning
######################

export TRAIN_FILE=datasets/cc/cc_train_6-0_no-symbols.txt
export TEST_FILE=datasets/cc/cc_dev_6-0_no-symbols.txt

for i in 0.25 0.5 2 4; do
python run_language_modeling.py \
    --output_dir ./models/parl-gpt2-v1_finetune_cc_v1_${i}e-4 \
    --model_type gpt2 \
    --config_name ./models/gpt2 \
    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
    --model_name_or_path ./models/parl-gpt2-v1 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs 5 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_train_batch_size 4 \
    --evaluate_during_training \
    --seed 42 \
    --learning_rate ${i}e-4
done
