
source asr-utils

#get-vocab-from-txt -t ../datasets/cc/cc_train_6-0_no-symbols.txt
#get-vocab-from-txt -oc -f ${i} -t train_txt/gpt2_generated_clean.txt
cat vocab/gpt2_generated_clean.stat | python rmv_tokens_found_in_arg.py \
vocab/cc_train_6-0_no-symbols_f0.voc > uniq_in_gpt2.txt  


#for i in {0..19}; do
#    get-vocab-from-txt -oc -f ${i} -t train_txt/gpt2_generated_clean.txt
#    cat vocab/gpt2_generated_clean_f${i}.voc | sort > vocab/gpt2_generated_clean_f${i}_sorted.voc
#    comm -1 -3 vocab/cc_train_6-0_no-symbols_f0_sorted.voc vocab/gpt2_generated_clean_f${i}_sorted.voc > uniq_in_gpt2_f${i}.txt    
#done
