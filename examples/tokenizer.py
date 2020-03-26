
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

#paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]
paths = ["datasets/cc/cc_train_6-0_no-symbols.txt"]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

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
tokenizer.train(files=paths, vocab_size=30_000, min_frequency=2)

# Save files to disk
tokenizer.save("models/cc-ByteLevelBPE-nospec")
