
from pathlib import Path

import sentencepiece as spm
from transformers import AutoTokenizer,XLNetTokenizer
from tokenizers import ByteLevelBPETokenizer,SentencePieceBPETokenizer

#paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]
paths = ["datasets/cc/cc_train_6-0_no-symbols.txt"]

##ByteLevelBPE
##############

# Initialize a tokenizer
#tokenizer = ByteLevelBPETokenizer()

##Roberta
## Customize training
#tokenizer.train(files=paths, vocab_size=30_000, min_frequency=2, special_tokens=[
#    "<s>",
#    "<pad>",
#    "</s>",
#    "<unk>",
#    "<mask>",
#])
#
## Save files to disk
#tokenizer.save("models/cc-ByteLevelBPE")

##GPT-2
# Customize training
#tokenizer.train(files=paths, vocab_size=30_000, min_frequency=2)

# Save files to disk
#tokenizer.save("models/cc-ByteLevelBPE-nospec")

##SentencePiece
###############

## XLnet tokenizer

## load a pretrained model
#tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
## save model to directory
#tokenizer.save_pretrained("models/cc-SentencePiece-nospec")

## train custom tokenizer
#tokenizer = SentencePieceBPETokenizer()
#tokenizer.train(files=paths, vocab_size=30_000, min_frequency=2)
## overwrite pretrained model with custom
#tokenizer.save("models/cc-SentencePiece-nospec")

## Train SentencePiece model
spm.SentencePieceTrainer.Train('--input=datasets/cc/cc_train_6-0_no-symbols.txt --model_prefix=models/cc-SentencePiece-xlnet/spiece --character_coverage=1.0 --vocab_size=32000 --model_type=bpe')



 #
 #
 #
 #
 #
 #
 #
 #
