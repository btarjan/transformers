
SAMPLED_TXT=datasets/cc/cc_train_6-0_no-symbols_no_empty_line.txt

for i in {1..30000}; do 
    ##Random int in range 1-6
    NUM_OF_WORDS=$(( 1 + RANDOM % 6 ))
    
    ##Sampling prompt from the training text
    PROMPT=$(shuf -n 1 ${SAMPLED_TXT} | cut -d ' ' -f 1-${NUM_OF_WORDS})
    #echo $PROMPT
    python ./run_generation.py \
        --model_type=gpt2 \
        --length=512 \
        --prompt="${PROMPT}" \
        --seed=${i} \
        --num_return_sequences=90 \
        --model_name_or_path=models/cc-gpt2-v4
done

