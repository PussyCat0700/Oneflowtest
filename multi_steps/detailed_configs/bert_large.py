from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification

# util functions TODO move elsewhere
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# params
task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "distilbert-base-uncased"
label_all_tokens = True
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
class MixDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    enable_oneflow:bool=False
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        if self.enable_oneflow:
            import oneflow as of
            batch = {k: of.tensor(v) for k, v in batch.items()}
        return batch
        
data_collator = MixDataCollatorForTokenClassification(tokenizer)
# dataset
datasets = load_dataset("conll2003")
label_list = datasets["train"].features[f"{task}_tags"].feature.names
num_labels = len(label_list)
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, remove_columns=['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])

# https://huggingface.co/distilbert/distilbert-base-uncased
cfg = dict(
    vocab_size=30522,
    hidden_size=768,
    hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    num_tokentypes=2,
    add_pooling_layer=True,
    initializer_range=0.02,
    layernorm_eps=1e-5,
    bias_gelu_fusion=True,
    bias_dropout_fusion=True,
    scale_mask_softmax_fusion=True,
    apply_query_key_layer_scaling=True,
    apply_residual_post_layernorm=False,
    add_binary_head=True,
    amp_enabled=False,
    num_labels=num_labels,
    classifier_dropout=0.1,
)