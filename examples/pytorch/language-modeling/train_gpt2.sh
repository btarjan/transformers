
# TODO:
# több epoch
# config nélküli futtatás
# tokenizer nélküli futtatás
# --max_grad_norm 5
# run_clm_no_trainer.py
# epoch 20
# --fp16
## --fp16_opt_level 01

train_gpt2() {
  python run_clm.py \
      --model_type gpt2 \
      --config_name ./configs/gpt2 \
      --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
      --output_dir ./models/${1}-gpt2-BS_${batch_size}-LR_${learning_rate}-epoch_${num_train_epochs} \
      --do_train \
      --train_file=${train_file} \
      --do_eval \
      --evaluation_strategy epoch \
      --validation_file=${test_file} \
      --num_train_epochs ${num_train_epochs} \
      --save_strategy epoch \
      --save_total_limit 2 \
      --per_device_train_batch_size ${batch_size} \
      --per_device_eval_batch_size ${batch_size} \
      --seed 42 \
      --learning_rate ${learning_rate}
}

# CC train

# datasets
train_file=../../datasets/cc/cc_train_6-0_no-symbols.txt
test_file=../../datasets/cc/cc_dev_6-0_no-symbols.txt

# training
batch_size=4
learning_rate=1e-4
num_train_epochs=10
train_gpt2 cc

batch_size=4
learning_rate=1e-4
num_train_epochs=20
train_gpt2 cc

batch_size=1
learning_rate=1e-4
num_train_epochs=10
train_gpt2 cc

batch_size=1
learning_rate=1e-4
num_train_epochs=20
train_gpt2 cc

# Parlament train

#train_file=../../datasets/mtva/MTVA-parlament_8-0_for_CC.txt
#test_file=../../datasets/cc/cc_dev_6-0_no-symbols.txt
#
#batch_size=4
#learning_rate=1e-4
#num_train_epochs=10
#train_gpt2 parl
