
export TRAIN_FILE=datasets/mtva/MTVA-parlament_8-0_for_CC.txt
export TEST_FILE=datasets/cc/cc_dev_6-0_no-symbols.txt


##Train GPT-2 with gpt2(small) config on CC data
################################################

export TRAIN_FILE=datasets/mtva/MTVA-parlament_8-0_for_CC.txt
#export TEST_FILE=datasets/mtva/MTVA-parlament_8-0_for_CC_dev.txt

BATCH_SIZE=4
GRAD_ACC=4

#python run_language_modeling.py \
#    --output_dir ./models/parl-gpt2-BS_${BATCH_SIZE}-GA_${GRAD_ACC} \
#    --model_type gpt2 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 10 \
#    --logging_steps 500 \
#    --save_steps 5000 \
#    --per_gpu_train_batch_size ${BATCH_SIZE} \
#    --evaluate_during_training \
#    --eval_all_checkpoints \
#    --seed 42 \
#    --gradient_accumulation_steps ${GRAD_ACC} \
#    --learning_rate 1e-4

BATCH_SIZE=1
GRAD_ACC=4

#python run_language_modeling.py \
#    --output_dir ./models/parl-gpt2-medium-2GPU-BS_${BATCH_SIZE}-GA_${GRAD_ACC} \
#    --model_type gpt2 \
#    --config_name ./models/gpt2-medium \
#    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 5 \
#    --logging_steps 500 \
#    --save_steps 5000 \
#    --per_gpu_train_batch_size ${BATCH_SIZE} \
#    --evaluate_during_training \
#    --eval_all_checkpoints \
#    --seed 42 \
#    --gradient_accumulation_steps ${GRAD_ACC} \
#    --learning_rate 1e-4

#python run_language_modeling.py \
#    --output_dir ./models/parl-gpt2-medium-v1 \
#    --model_type gpt2 \
#    --config_name ./models/gpt2-medium \
#    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 10 \
#    --logging_steps 2000 \
#    --save_steps 20000 \
#    --per_gpu_train_batch_size 1 \
#    --evaluate_during_training \
#    --eval_all_checkpoints \
#    --seed 42 \
#    --learning_rate 1e-4

## Reconstruction of old results

BATCH_SIZE=1
GRAD_ACC=1

python run_language_modeling.py \
    --output_dir ./models/parl-gpt2-BS_${BATCH_SIZE}-GA_${GRAD_ACC}_rerun \
    --model_type gpt2 \
    --config_name ./models/gpt2 \
    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs 10 \
    --logging_steps 500 \
    --save_steps 5000 \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --evaluate_during_training \
    --eval_all_checkpoints \
    --seed 42 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --learning_rate 1e-4

