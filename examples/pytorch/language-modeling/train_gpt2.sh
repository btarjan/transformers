
##Run wikitext sample LM training
################################# 

#export TRAIN_FILE=datasets/wikitext-2-raw/wiki.train.raw
#export TEST_FILE=datasets/wikitext-2-raw/wiki.test.raw
#
#python run_language_modeling.py \
#    --output_dir=output \
#    --model_type=gpt2 \
#    --model_name_or_path=gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --per_gpu_train_batch_size 1 \
#    --per_gpu_eval_batch_size 1

##Train Roberta with EsperBERTo config on CC data
#################################################

export TRAIN_FILE=datasets/cc/cc_train_6-0_no-symbols.txt
export TEST_FILE=datasets/cc/cc_dev_6-0_no-symbols.txt

#python run_language_modeling.py \
#    --output_dir ./models/cc-roberta-v1 \
#    --model_type roberta \
#    --mlm \
#    --config_name ./models/EsperBERTo-small \
#    --tokenizer_name ./models/cc-ByteLevelBPE \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 50 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 8 \
#    --evaluate_during_training \
#    --seed 42 \
#    --overwrite_output_dir \
#    --learning_rate 1e-4

##PPL Eval
#export TEST_EVAL_FILE=datasets/cc/cc_eval_6-0_no-symbols.txt
#
#python run_language_modeling.py \
#    --output_dir ./models/cc-roberta-v1 \
#    --model_type roberta \
#    --model_name_or_path=./models/cc-roberta-v1 \
#    --mlm \
#    --config_name ./models/cc-roberta-v1 \
#    --tokenizer_name ./models/cc-roberta-v1 \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_EVAL_FILE \
#    --per_gpu_eval_batch_size 4


##Train GPT-2 with gpt2(small) config on CC data
################################################

export TRAIN_FILE=datasets/cc/cc_train_6-0_no-symbols.txt
export TEST_FILE=datasets/cc/cc_dev_6-0_no-symbols.txt

#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-v1 \
#    --model_type gpt2 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-nospec \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 50 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 4 \
#    --evaluate_during_training \
#    --seed 42
#    --overwrite_output_dir \
#    --learning_rate 1e-4

#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-v2 \
#    --model_type gpt2 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-nospec \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 20 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 4 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 1e-4
#
#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-v3 \
#    --model_type gpt2 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 20 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 4 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 1e-4

#for i in 1 4 16 64 256; do
#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-v4-grad_acc_${i} \
#    --model_type gpt2 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 10 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 2 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 1e-4 \
#    --gradient_accumulation_steps ${i}
#done

#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-v6 \
#    --model_type gpt2 \
#    --config_name ./models/gpt2-reg_05 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-nospec \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 20 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 4 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 1e-4

#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-medium-v1 \
#    --model_type gpt2 \
#    --config_name ./models/gpt2-medium \
#    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 10 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 1 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 1e-4

#for i in 6 8 10 12; do
#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-lr_3e4-epoch_${i}-2GPU \
#    --model_type gpt2 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs ${i} \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 4 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 3e-4
#done
#
#for i in 6 8 10 12; do
#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-lr_4e4-epoch_${i}-2GPU \
#    --model_type gpt2 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs ${i} \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 4 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 4e-4
#done



#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-v8 \
#    --model_type gpt2 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-nospec \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 20 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 8 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 1e-4

#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-small-v1 \
#    --model_type gpt2 \
#    --config_name ./models/gpt2-small \
#    --tokenizer_name ./models/cc-ByteLevelBPE-nospec \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 10 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 8 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 1e-4

#python run_language_modeling.py \
#    --output_dir ./models/cc-gpt2-v9 \
#    --model_type gpt2 \
#    --config_name ./models/gpt2 \
#    --tokenizer_name ./models/cc-ByteLevelBPE-nospec \
#    --do_train \
#    --train_data_file=$TRAIN_FILE \
#    --do_eval \
#    --eval_data_file=$TEST_FILE \
#    --num_train_epochs 10 \
#    --save_total_limit 2 \
#    --save_steps 2000 \
#    --per_gpu_train_batch_size 8 \
#    --evaluate_during_training \
#    --seed 42 \
#    --learning_rate 1e-4


## Reconstruction of old results

export TRAIN_FILE=../../datasets/cc/cc_train_6-0_no-symbols.txt
export TEST_FILE=../../datasets/cc/cc_dev_6-0_no-symbols.txt

python run_clm.py \
    --output_dir ./models/cc-gpt2-v4_rerun2 \
    --model_type gpt2 \
    --config_name ./models/gpt2 \
    --tokenizer_name ./models/cc-ByteLevelBPE-gpt2 \
    --do_train \
    --train_file=$TRAIN_FILE \
    --do_eval \
    --validation_file=$TEST_FILE \
    --num_train_epochs 10 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_train_batch_size 4 \
    --seed 42 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir \
