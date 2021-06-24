
# TODO:
# több epoch
# config nélküli futtatás
# tokenizer nélküli futtatás
# --max_grad_norm 5
# run_clm_no_trainer.py
# epoch 20
# --fp16
## --fp16_opt_level 01

# datasets
export TRAIN_FILE=../../datasets/cc/cc_train_6-0_no-symbols.txt
export TEST_FILE=../../datasets/cc/cc_dev_6-0_no-symbols.txt

# max_grad_norm 5
batch_size=4
learning_rate=1e-4

python run_clm.py \
    --model_type gpt2 \
    --config_name ./configs/gpt2 \
    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
    --output_dir ./models/cc-gpt2-BS_${batch_size}-LR_${learning_rate}-MG_5 \
    --do_train \
    --train_file=$TRAIN_FILE \
    --do_eval \
    --evaluation_strategy epoch \
    --validation_file=$TEST_FILE \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --seed 42 \
    --learning_rate ${learning_rate} \
    --max_grad_norm 5


# run_clm_no_trainer
batch_size=4
learning_rate=1e-4

python run_clm_no_trainer.py \
    --model_type gpt2 \
    --config_name ./configs/gpt2 \
    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
    --output_dir ./models/cc-gpt2-BS_${batch_size}-LR_${learning_rate}-no_trainer \
    --do_train \
    --train_file=$TRAIN_FILE \
    --do_eval \
    --evaluation_strategy epoch \
    --validation_file=$TEST_FILE \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --seed 42 \
    --learning_rate ${learning_rate} \


# --num_train_epochs 20
batch_size=4
learning_rate=1e-4

python run_clm.py \
    --model_type gpt2 \
    --config_name ./configs/gpt2 \
    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
    --output_dir ./models/cc-gpt2-BS_${batch_size}-LR_${learning_rate} \
    --do_train \
    --train_file=$TRAIN_FILE \
    --do_eval \
    --evaluation_strategy epoch \
    --validation_file=$TEST_FILE \
    --num_train_epochs 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --seed 42 \
    --learning_rate ${learning_rate} \


# batch_size=1
batch_size=1
learning_rate=1e-4

python run_clm.py \
    --model_type gpt2 \
    --config_name ./configs/gpt2 \
    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
    --output_dir ./models/cc-gpt2-BS_${batch_size}-LR_${learning_rate} \
    --do_train \
    --train_file=$TRAIN_FILE \
    --do_eval \
    --evaluation_strategy epoch \
    --validation_file=$TEST_FILE \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --seed 42 \
    --learning_rate ${learning_rate} \
