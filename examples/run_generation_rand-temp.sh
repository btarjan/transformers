
SAMPLED_TXT=datasets/cc/cc_train_6-0_no-symbols_no_empty_line.txt

for i in {1..30000}; do 
    ##Random int in range 1-6
    NUM_OF_WORDS=$(( 1 + RANDOM % 6 ))
    
    ##Random int in range 0-5
    TEMP=$(( NUM_OF_WORDS - 1 ))

    ##Sampling prompt from the training text
    PROMPT=$(shuf -n 1 ${SAMPLED_TXT} | cut -d ' ' -f 1-${NUM_OF_WORDS})
    #echo $PROMPT
    python ./run_generation.py \
        --model_type=gpt2 \
        --length=512 \
        --prompt="${PROMPT}" \
        --seed=${i} \
        --temperature 1.${TEMP} \
        --num_return_sequences=90 \
        --model_name_or_path=models/parl-gpt2-v1_finetune_cc_v1_epoch-4
done

