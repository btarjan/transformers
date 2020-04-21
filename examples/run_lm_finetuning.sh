
## CC GPT-2 finetuning
######################

export TRAIN_FILE=datasets/cc/cc_train_6-0_no-symbols.txt
export TEST_FILE=datasets/cc/cc_dev_6-0_no-symbols.txt

#parl-gpt2-BS_4-GA_4

PRETRAINED=parl-gpt2-BS_4-GA_4
BATCH_SIZE=4

for i in 2 3 4 5 6; do
python run_language_modeling.py \
    --output_dir ./models/${PRETRAINED}_finetune_cc_epoch-${i} \
    --model_type gpt2 \
    --config_name ./models/gpt2 \
    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
    --model_name_or_path ./models/${PRETRAINED} \
    --do_train \
    --train_data_file=${TRAIN_FILE} \
    --do_eval \
    --eval_data_file=${TEST_FILE} \
    --num_train_epochs ${i} \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --evaluate_during_training \
    --seed 42 \
    --learning_rate 1e-4
done

##PPL Eval
#export TEST_EVAL_FILE=datasets/cc/cc_eval_6-0_no-symbols.txt
#
#python run_language_modeling.py \
#    --output_dir ./models/parl-gpt2-v1_finetune_cc_v1_epoch-4_EVAL \
#    --model_type gpt2 \
#    --model_name_or_path ./models/parl-gpt2-v1_finetune_cc_v1_epoch-4 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_EVAL_FILE \
#    --per_gpu_eval_batch_size 4

