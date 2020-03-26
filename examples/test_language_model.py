
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./models/cc-roberta-v1",
    tokenizer="./models/cc-roberta-v1"
)

# The sun <mask>.
# =>

result = fill_mask("jó napot <mask>")
print(result)
