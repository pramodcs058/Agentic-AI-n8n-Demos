# Lab 6 – Transformer Token Attribution with DistilBERT and Captum

**Platform:** Google Colab  
**Level:** Undergraduate students / beginner to intermediate  
**Estimated duration:** 60–75 minutes  
**Dataset:** GLUE – Stanford Sentiment Treebank (SST-2)  
**Pretrained model:** DistilBERT fine-tuned on SST-2  
**Libraries:** PyTorch, Hugging Face Transformers, Hugging Face Datasets, Captum, Matplotlib  
**Explainability method:** Layer Integrated Gradients (LayerIG)  
**Task:** Explain a positive/negative sentiment prediction by attributing the selected class score to individual input tokens.

---

## 1. Lab overview

### Learning objectives

By the end of this lab, students will be able to:

1. Explain how a Transformer model performs text classification.
2. Load a public sentiment dataset and a pretrained DistilBERT classifier.
3. Tokenize a sentence into model-specific WordPiece tokens.
4. Obtain the predicted sentiment and class probabilities.
5. Explain why gradients cannot be calculated directly with respect to integer token IDs.
6. Use Captum `LayerIntegratedGradients` to attribute a class score to token embeddings.
7. Visualise positive and negative token attributions.
8. Verify the Integrated Gradients completeness property.
9. Explain why token attribution is not the same as attention visualisation or causal proof.

### Case study: “Which words influenced the sentiment prediction?”

Imagine a product-review application. A customer writes:

> “The camera is excellent, but the battery life is disappointing.”

The application predicts whether the overall sentiment is positive or negative. A reviewer then asks:

> “Which words influenced the model's prediction?”

Token attribution assigns an attribution score to each token in the input sentence. A positive score means the token pushes the selected class score upward relative to the reference input. A negative score means it pushes that score downward.

For this lab, we use a pretrained DistilBERT model fine-tuned for binary sentiment classification. Students do not need to train a Transformer from scratch; the focus is on explaining its prediction.

### Dataset and model

We use:

- **Dataset:** GLUE SST-2, a sentence-level sentiment dataset containing movie-review sentences labelled negative or positive.
- **Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, a DistilBERT checkpoint fine-tuned on SST-2.

The SST-2 dataset provides sentences and their sentiment labels. The model is already fine-tuned on this task. The notebook uses the validation split to choose and inspect an example. It does not retrain the Transformer.

### What students will build

- A working DistilBERT sentiment classifier.
- A prediction for a sentence from the SST-2 validation set.
- A reference input constructed using `[PAD]` tokens while retaining `[CLS]` and `[SEP]`.
- Token-level attribution scores using Captum Layer Integrated Gradients.
- A coloured token visualisation.
- A completeness check comparing the sum of token attributions with the difference between the selected class logit for the actual and reference inputs.
- An optional comparison explaining the opposite sentiment class.

---

## 2. Open Google Colab

1. Open [https://colab.research.google.com/](https://colab.research.google.com/).
2. Select **New notebook**.
3. Rename it `Lab-6-Transformer-Token-Attribution`.
4. Select **Runtime → Change runtime type**.
5. Choose **Python 3**.
6. A CPU runtime is sufficient for one-sentence attribution. A GPU can make the attribution calculation faster, but it is not mandatory.
7. Run the cells in order.

The dataset and model are downloaded from Hugging Face, so the Colab runtime needs an internet connection. The first model download may take a few minutes.

---

## 3. Install and import the libraries

### Cell 1 – Install dependencies

```python
%pip install -q torch transformers datasets captum matplotlib pandas
```

Colab normally includes PyTorch. The command ensures the required packages are available. If Colab requests a runtime restart, restart the session and then rerun the import cell.

### Cell 2 – Import packages

```python
import html
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from captum.attr import LayerIntegratedGradients

from IPython.display import HTML, display

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("Device:", device)
print("Imports completed.")
```

**Instructor note:** PyTorch executes the neural network. Transformers supplies the pretrained DistilBERT model and tokenizer. Datasets downloads SST-2. Captum calculates attributions for the model's embedding layer.

---

## 4. Load the SST-2 dataset

SST-2 is a binary sentiment classification task. Its labels are:

- `0`: negative
- `1`: positive

We use the current Hugging Face Hub dataset repository `nyu-mll/glue`, with the `sst2` configuration.

### Cell 3 – Load the validation split

```python
sst2_validation = load_dataset(
    "nyu-mll/glue",
    "sst2",
    split="validation"
)

print("Validation examples:", len(sst2_validation))
print("Columns:", sst2_validation.column_names)

display(pd.DataFrame(sst2_validation[:5]))
```

### Cell 4 – Inspect the examples and labels

```python
label_names = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

for index in range(5):
    row = sst2_validation[index]
    print(f"Example {index}")
    print("Sentence:", row["sentence"])
    print("Gold label:", label_names[int(row["label"])])
    print("-" * 70)
```

**Discuss with students:**

- What is the input to a sentiment classifier?
- What is the output?
- Why is sentiment a classification problem rather than a regression problem?
- Why do human-labelled datasets sometimes contain ambiguous sentences?

---

## 5. Load the pretrained DistilBERT model

The selected model is already fine-tuned on SST-2. DistilBERT is a smaller, distilled Transformer model based on BERT. We use it so the class can focus on token attribution instead of spending the session training a large language model.

### Cell 5 – Load tokenizer and model

```python
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

print("Model loaded:", MODEL_NAME)
print("Number of labels:", model.config.num_labels)
print("Model label mapping:", model.config.id2label)
```

### Cell 6 – Confirm the sentiment class IDs

The SST-2 checkpoint normally maps class `0` to negative and class `1` to positive. We inspect the model configuration and use the SST-2 mapping explicitly for the teaching example.

```python
id2label = {
    int(key): str(value)
    for key, value in model.config.id2label.items()
}

print("Configured labels:", id2label)

# SST-2 class convention:
# 0 = negative, 1 = positive
NEGATIVE_CLASS_ID = 0
POSITIVE_CLASS_ID = 1

assert model.config.num_labels == 2, "This lab expects a binary sentiment classifier."

print("Negative class ID:", NEGATIVE_CLASS_ID)
print("Positive class ID:", POSITIVE_CLASS_ID)
```

**Instructor note:** A model's output is a pair of logits. Softmax converts the logits into probabilities that sum to 1. The predicted class is the class with the highest logit/probability.

---

## 6. Evaluate the model on a small validation sample

We will inspect a small subset rather than running a full benchmark. This is enough to show that the pretrained classifier is performing the intended task before explaining one prediction.

### Cell 7 – Run predictions on 128 validation sentences

```python
EVAL_COUNT = min(128, len(sst2_validation))

eval_rows = sst2_validation.select(range(EVAL_COUNT))
eval_texts = eval_rows["sentence"]
eval_labels = np.asarray(eval_rows["label"], dtype=int)

predicted_labels = []
predicted_confidences = []

BATCH_SIZE = 16

for start in range(0, EVAL_COUNT, BATCH_SIZE):
    batch_texts = eval_texts[start:start + BATCH_SIZE]

    batch_inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**batch_inputs).logits
        probabilities = torch.softmax(logits, dim=-1)

    batch_predictions = torch.argmax(probabilities, dim=-1)
    batch_confidences = torch.max(probabilities, dim=-1).values

    predicted_labels.extend(batch_predictions.cpu().numpy().tolist())
    predicted_confidences.extend(batch_confidences.cpu().numpy().tolist())

predicted_labels = np.asarray(predicted_labels)
predicted_confidences = np.asarray(predicted_confidences)

subset_accuracy = float(np.mean(predicted_labels == eval_labels))

print(f"Accuracy on this {EVAL_COUNT}-sentence validation subset: {subset_accuracy:.3f}")
```

This is a small demonstration subset, not an official benchmark result.

### Cell 8 – View a few model predictions

```python
prediction_preview = pd.DataFrame({
    "sentence": eval_texts[:10],
    "actual_label": [label_names[int(x)] for x in eval_labels[:10]],
    "predicted_label": [label_names[int(x)] for x in predicted_labels[:10]],
    "confidence": predicted_confidences[:10]
})

display(prediction_preview)
```

---

## 7. Select a sentence to explain

For a useful classroom example, we select a sentence that:
- Has enough words to make token attribution interesting.
- Is predicted correctly on the validation sample.
- Has reasonably high model confidence.

If no sentence meets the confidence rule, the code selects the highest-confidence example from the sample.

### Cell 9 – Select the example automatically

```python
candidate_positions = []

for index, sentence in enumerate(eval_texts):
    word_count = len(str(sentence).split())

    is_correct = predicted_labels[index] == eval_labels[index]
    is_confident = predicted_confidences[index] >= 0.80
    is_reasonable_length = 6 <= word_count <= 30

    if is_correct and is_confident and is_reasonable_length:
        candidate_positions.append(index)

if candidate_positions:
    selected_position = candidate_positions[len(candidate_positions) // 2]
    selection_note = "correct prediction with confidence >= 0.80"
else:
    selected_position = int(np.argmax(predicted_confidences))
    selection_note = "highest-confidence example in the inspected subset"

selected_text = str(eval_texts[selected_position])
selected_gold_label = int(eval_labels[selected_position])

print("Selection method:", selection_note)
print("Sentence:", selected_text)
print("Dataset label:", label_names[selected_gold_label])
```

### Cell 10 – You may also use your own sentence

For a live classroom demonstration, replace `selected_text` below with a sentence of your choice. Start with a short English review sentence.

```python
# Example alternatives:
# selected_text = "The movie was entertaining, funny, and beautifully acted."
# selected_text = "The story was dull and the ending was disappointing."
# selected_text = "The visuals were impressive, but the plot was weak."

# Keep the sentence reasonably short for this demonstration.
print("Sentence selected for explanation:")
print(selected_text)
```

**Teaching prompt:** Ask students to predict the sentiment before running the model. Then compare their interpretation with the model's output.

---

## 8. Tokenize the selected sentence

Transformers do not directly process ordinary text. The tokenizer converts the text into token IDs from the model's vocabulary.

DistilBERT uses WordPiece tokenization. A word can become more than one token. For example, an uncommon word may be split into a first piece and one or more continuation pieces that begin with `##`.

We use a maximum sequence length of 64 for this short example.

### Cell 11 – Tokenize the sentence

```python
MAX_LENGTH = 64

encoded = tokenizer(
    selected_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH,
    padding="max_length"
)

input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

print("Input IDs shape:", tuple(input_ids.shape))
print("Attention mask shape:", tuple(attention_mask.shape))

token_ids = input_ids[0].detach().cpu().tolist()
tokens = tokenizer.convert_ids_to_tokens(token_ids)

print("\nTokens:")
print(tokens)
```

### Cell 12 – Inspect tokens, token IDs, and attention mask

```python
token_inspection = pd.DataFrame({
    "position": range(input_ids.shape[1]),
    "token": tokens,
    "token_id": token_ids,
    "attention_mask": attention_mask[0].detach().cpu().tolist()
})

display(token_inspection.head(30))
```

**Explain the special tokens:**

- `[CLS]`: a classification token placed at the beginning of the sequence.
- `[SEP]`: a separator/end marker.
- `[PAD]`: padding used to make sequences the same length.
- WordPiece fragments such as `##ful`: continuation pieces of a word.

The model uses the attention mask to distinguish actual tokens from padding positions.

---

## 9. Obtain the original model prediction

### Cell 13 – Predict the sentiment

```python
model.eval()

with torch.no_grad():
    original_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).logits

    original_probabilities = torch.softmax(original_logits, dim=-1)

predicted_class_id = int(torch.argmax(original_logits, dim=-1).item())

probability_negative = float(original_probabilities[0, NEGATIVE_CLASS_ID].item())
probability_positive = float(original_probabilities[0, POSITIVE_CLASS_ID].item())

print("Sentence:", selected_text)
print("Predicted sentiment:", label_names[predicted_class_id])
print(f"Negative probability: {probability_negative:.4f}")
print(f"Positive probability: {probability_positive:.4f}")
```

### Cell 14 – Plot the sentiment probabilities

```python
plt.figure(figsize=(6, 4))

bars = plt.bar(
    ["Negative", "Positive"],
    [probability_negative, probability_positive]
)

plt.ylim(0, 1)
plt.ylabel("Predicted probability")
plt.title("DistilBERT Sentiment Prediction")

for bar, value in zip(bars, [probability_negative, probability_positive]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.show()
```

**Instructor explanation:** We will now explain the score for the predicted class. If the model predicts positive, the attribution calculation targets the positive logit. If it predicts negative, it targets the negative logit.

---

## 10. Why we need Layer Integrated Gradients

Token IDs are integers. A gradient-based method cannot take a meaningful derivative directly with respect to an integer token ID.

For example, token ID `100` is not numerically halfway between token ID `80` and token ID `120` in a meaningful language sense.

Instead, a Transformer converts token IDs into continuous embedding vectors. We can calculate gradients with respect to these embedding representations.

Captum's `LayerIntegratedGradients` is designed for this kind of situation. We identify the model's embedding layer:

```python
model.distilbert.embeddings
```

We then attribute the selected class score to the output of that embedding layer and aggregate the embedding dimensions to obtain one attribution score per token.

### Integrated Gradients idea

For input \(x\), reference input \(x'\), and model score \(F\), Integrated Gradients calculates:

\[
IG_i(x) =
(x_i - x'_i)
\int_0^1
\frac{\partial F(x' + \alpha(x-x'))}{\partial x_i}
\,d\alpha
\]

Here:
- The actual input is the selected sentence represented by token IDs.
- The embedding layer converts token IDs into continuous vectors.
- The reference input uses padding tokens in place of ordinary tokens.
- The target score is the logit for one sentiment class.

We use a reference sentence with the same sequence length and special-token positions. Ordinary token positions are replaced by `[PAD]`. We keep the attention mask fixed so that the attribution comparison is made for the same sequence positions.

---

## 11. Construct the reference input

### Cell 15 – Create a PAD-based reference sequence

The original token sequence has this general structure:

`[CLS] token token token ... [SEP] [PAD] [PAD] ...`

Our reference sequence is:

`[CLS] [PAD] [PAD] [PAD] ... [SEP] [PAD] [PAD] ...`

The `[CLS]` and `[SEP]` tokens remain in their original positions. The ordinary token positions are replaced with the tokenizer's padding token ID.

```python
if tokenizer.pad_token_id is None:
    raise ValueError("Tokenizer does not define a PAD token.")

baseline_ids = torch.full_like(
    input_ids,
    fill_value=tokenizer.pad_token_id
)

# Retain [CLS] at the first position.
baseline_ids[:, 0] = tokenizer.cls_token_id

# Find the final non-padding position, which is [SEP] for this single sentence.
valid_length = int(attention_mask[0].sum().item())
sep_position = valid_length - 1

if sep_position < 1:
    raise ValueError("Unexpected tokenized sequence: [SEP] position was not found.")

baseline_ids[:, sep_position] = tokenizer.sep_token_id

print("Input tokens:")
print(tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist()))

print("\nReference tokens:")
print(tokenizer.convert_ids_to_tokens(baseline_ids[0].detach().cpu().tolist()))

print("\nAttention mask is retained from the original input:")
print(attention_mask[0].detach().cpu().tolist())
```

**Important interpretation:** The baseline is not an empty sentence in a linguistic sense. It is a reference embedding sequence used by the attribution calculation. Token attribution always depends on the reference chosen.

---

## 12. Define a forward function for the target sentiment class

Captum needs a function that returns one scalar score per example. We create a function that returns the selected class logit.

We use the **logit**, not the softmax probability, as the attribution target. This is a common way to explain a specific model output without introducing the additional interactions created by the softmax normalisation.

### Cell 16 – Define the target forward function

```python
TARGET_CLASS_ID = predicted_class_id

def forward_for_target_class(input_ids_arg, attention_mask_arg):
    """
    Return the logit for the selected target class.

    Inputs:
      input_ids_arg: integer token IDs, shape [batch, sequence_length]
      attention_mask_arg: attention mask, same batch and sequence length

    Output:
      one scalar logit per example, shape [batch]
    """
    outputs = model(
        input_ids=input_ids_arg,
        attention_mask=attention_mask_arg
    )

    return outputs.logits[:, TARGET_CLASS_ID]


print("Target class:", label_names[TARGET_CLASS_ID])
```

---

## 13. Calculate token attributions with Captum

### Cell 17 – Create the Layer Integrated Gradients object

For this DistilBERT sequence-classification model, the embedding layer is available at:

`model.distilbert.embeddings`

```python
layer_ig = LayerIntegratedGradients(
    forward_for_target_class,
    model.distilbert.embeddings
)

print("LayerIntegratedGradients object created.")
```

### Cell 18 – Calculate the attributions

```python
model.eval()

attributions, convergence_delta = layer_ig.attribute(
    inputs=input_ids,
    baselines=baseline_ids,
    additional_forward_args=(attention_mask,),
    n_steps=50,
    method="gausslegendre",
    return_convergence_delta=True
)

print("Attribution tensor shape:", tuple(attributions.shape))
print(
    "Convergence delta:",
    float(convergence_delta.detach().cpu().reshape(-1)[0].item())
)
```

The attribution tensor contains a vector for each token. The vector length equals the embedding dimension. We next sum the embedding dimensions to get one signed score per token.

**Instructor note:**

- `n_steps=50` controls the numerical approximation along the path from baseline to input.
- `method="gausslegendre"` selects the numerical integration method.
- `additional_forward_args` passes the attention mask to the model for each attribution evaluation.
- `convergence_delta` is a completeness-related approximation check, not a sentiment accuracy score.

---

## 14. Aggregate embedding attributions into token scores

### Cell 19 – Sum the embedding dimensions

```python
# Shape before aggregation: [batch_size, sequence_length, hidden_dimension]
# Sum the embedding dimension to get one score for each token.
token_attributions = (
    attributions
    .sum(dim=-1)
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
)

token_ids_cpu = input_ids[0].detach().cpu().tolist()
tokens = tokenizer.convert_ids_to_tokens(token_ids_cpu)
attention_values = attention_mask[0].detach().cpu().tolist()

print("Number of tokens:", len(tokens))
print("Number of token scores:", len(token_attributions))
```

### Cell 20 – Create a token attribution table

```python
token_attribution_df = pd.DataFrame({
    "position": range(len(tokens)),
    "token": tokens,
    "attribution": token_attributions,
    "attention_mask": attention_values
})

# Flag special tokens for display decisions.
special_tokens = {
    tokenizer.cls_token,
    tokenizer.sep_token,
    tokenizer.pad_token
}

token_attribution_df["is_special_token"] = (
    token_attribution_df["token"].isin(special_tokens)
)

display(token_attribution_df.head(30))
```

A positive score pushes the selected class logit upward relative to the baseline. A negative score pushes it downward. The sign is meaningful only in relation to the selected target class and baseline.

---

## 15. Display the most influential tokens

We sort by absolute attribution to find the tokens with the largest magnitude. This list can include both positive and negative scores.

### Cell 21 – Show top tokens

```python
visible_tokens_df = token_attribution_df[
    (token_attribution_df["attention_mask"] == 1)
    & (~token_attribution_df["is_special_token"])
].copy()

top_tokens = visible_tokens_df.reindex(
    visible_tokens_df["attribution"].abs().sort_values(
        ascending=False
    ).index
).head(10)

display(top_tokens[["position", "token", "attribution"]])
```

**Ask students:**
- Which tokens have the largest positive attribution?
- Which tokens have the largest negative attribution?
- Do these tokens match the students' own interpretation of the sentence?
- Are any punctuation marks or subword fragments receiving attribution?

---

## 16. Visualise the attribution scores

### Cell 22 – Plot signed token attribution

```python
plot_df = visible_tokens_df.copy()

plt.figure(figsize=(max(10, len(plot_df) * 0.45), 5))

bar_colors = [
    "seagreen" if score >= 0 else "indianred"
    for score in plot_df["attribution"]
]

plt.bar(
    plot_df["token"],
    plot_df["attribution"],
    color=bar_colors
)

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Input token")
plt.ylabel("Integrated Gradients attribution")
plt.title(
    f"Token Attribution for the {label_names[TARGET_CLASS_ID]} Class"
)
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()
```

### How to explain the chart

- Green/positive bars increase the target class logit relative to the baseline.
- Red/negative bars decrease the target class logit relative to the baseline.
- A larger absolute score indicates a larger attribution magnitude for this specific input, target, model, and baseline.
- A token with a negative score is not automatically a “bad” word. It is evidence that the token pushes against the selected class score in this particular context.

**Important:** If the selected class is `NEGATIVE`, a positive attribution pushes the negative logit upward. It does not mean the token has a positive sentiment.

---

## 17. Create a coloured token view

The next cell gives a classroom-friendly view where each token is shown as a coloured label.

- Green background: positive attribution for the selected class.
- Red background: negative attribution for the selected class.
- Stronger colour: larger absolute attribution.
- The numeric value is printed next to each token.

Subword pieces beginning with `##` are displayed without the prefix so they are easier to read. Each piece is still shown as its own token.

### Cell 23 – Render highlighted tokens

```python
def attribution_to_style(score, max_abs_score):
    """
    Return a background colour based on the sign and magnitude
    of an attribution score.
    """
    if max_abs_score <= 0:
        strength = 0.0
    else:
        strength = min(abs(float(score)) / max_abs_score, 1.0)

    alpha = 0.12 + 0.60 * strength

    if score >= 0:
        return f"rgba(34, 139, 34, {alpha:.3f})"

    return f"rgba(220, 20, 60, {alpha:.3f})"


max_abs_score = float(
    np.max(np.abs(visible_tokens_df["attribution"].to_numpy()))
) if len(visible_tokens_df) else 0.0

token_spans = []

for _, row in visible_tokens_df.iterrows():
    token = str(row["token"])
    score = float(row["attribution"])

    # Remove the WordPiece continuation marker for display only.
    display_token = token[2:] if token.startswith("##") else token

    background = attribution_to_style(score, max_abs_score)

    token_spans.append(
        f"""
        <span style="
            display:inline-block;
            margin:4px;
            padding:7px 9px;
            border-radius:7px;
            background:{background};
            border:1px solid #999;
            font-family:monospace;
        ">
            {html.escape(display_token)}
            <br>
            <small>{score:+.3f}</small>
        </span>
        """
    )

html_output = """
<div style="line-height:2.5">
    <b>Token attribution for target class: """
html_output += html.escape(label_names[TARGET_CLASS_ID])
html_output += """</b><br>
"""
html_output += "\n".join(token_spans)
html_output += "</div>"

display(HTML(html_output))
```

**Instructor note:** Tell students that a word may be split into multiple WordPiece tokens. Token-level attribution is not necessarily identical to word-level attribution. For a production interface, subword scores can be aggregated back into complete words.

---

## 18. Verify the Integrated Gradients completeness property

Integrated Gradients is designed so that the sum of attributions approximates the difference between the model output for the actual input and the model output for the baseline.

\[
\sum_i IG_i(x) \approx F(x) - F(x')
\]

We calculate the two sides and compare them.

### Cell 24 – Calculate output difference and attribution sum

```python
model.eval()

with torch.no_grad():
    output_for_input = forward_for_target_class(
        input_ids,
        attention_mask
    )

    output_for_baseline = forward_for_target_class(
        baseline_ids,
        attention_mask
    )

attribution_sum = attributions.detach().sum()
output_difference = output_for_input - output_for_baseline

completeness_error = torch.abs(
    attribution_sum - output_difference
)

print(
    "Target logit for actual input:",
    float(output_for_input.item())
)

print(
    "Target logit for reference input:",
    float(output_for_baseline.item())
)

print(
    "Difference in target logits:",
    float(output_difference.item())
)

print(
    "Sum of token/embedding attributions:",
    float(attribution_sum.item())
)

print(
    "Absolute completeness error:",
    float(completeness_error.item())
)

print(
    "Captum convergence delta:",
    float(convergence_delta.detach().cpu().reshape(-1)[0].item())
)
```

The values are in the selected class's **logit units**, not probability units.

**Teaching note:** The convergence delta is a numerical approximation diagnostic. A small delta supports the numerical completeness check, but does not prove that the model is unbiased or that the attribution is a causal explanation.

---

## 19. Compare attributions for the opposite sentiment class

So far, we have explained the class predicted by the model. We can also calculate attributions for the opposite class.

For example:
- If the model predicts positive, calculate attributions for the negative logit.
- If the model predicts negative, calculate attributions for the positive logit.

The tokens that support one class may oppose the other class. However, because the model has two logits, the two explanation patterns should be interpreted as separate target-score explanations.

### Cell 25 – Define the opposite-class forward function

```python
OTHER_CLASS_ID = (
    NEGATIVE_CLASS_ID
    if TARGET_CLASS_ID == POSITIVE_CLASS_ID
    else POSITIVE_CLASS_ID
)

def forward_for_other_class(input_ids_arg, attention_mask_arg):
    outputs = model(
        input_ids=input_ids_arg,
        attention_mask=attention_mask_arg
    )

    return outputs.logits[:, OTHER_CLASS_ID]


print("Original target:", label_names[TARGET_CLASS_ID])
print("Other target:", label_names[OTHER_CLASS_ID])
```

### Cell 26 – Calculate opposite-class attributions

```python
other_class_lig = LayerIntegratedGradients(
    forward_for_other_class,
    model.distilbert.embeddings
)

other_attributions, other_delta = other_class_lig.attribute(
    inputs=input_ids,
    baselines=baseline_ids,
    additional_forward_args=(attention_mask,),
    n_steps=50,
    method="gausslegendre",
    return_convergence_delta=True
)

other_token_scores = (
    other_attributions
    .sum(dim=-1)
    .squeeze(0)
    .detach()
    .cpu()
    .numpy()
)

print(
    "Opposite-class convergence delta:",
    float(other_delta.detach().cpu().reshape(-1)[0].item())
)
```

### Cell 27 – Compare the two target classes

```python
comparison_df = pd.DataFrame({
    "token": tokens,
    f"{label_names[TARGET_CLASS_ID]}_attribution": token_attributions,
    f"{label_names[OTHER_CLASS_ID]}_attribution": other_token_scores,
    "attention_mask": attention_values
})

comparison_df = comparison_df[
    (comparison_df["attention_mask"] == 1)
    & (~comparison_df["token"].isin(special_tokens))
]

display(comparison_df)
```

**Discuss with students:**

- Does the token that most supports the predicted class also strongly oppose the other class?
- Are the two attribution patterns exact negatives of one another?
- Why should we always state which output class is being explained?

---

## 20. Experiment: change the reference input

The reference input influences the meaning of the explanation. In this lab, the baseline keeps `[CLS]` and `[SEP]` but replaces ordinary tokens with `[PAD]`.

Ask students to consider what would happen if the reference changed. A different reference could represent a different comparison question and may produce different attributions.

For this beginner lab, do not interpret the reference as a meaningful natural-language sentence. It is a numerical reference for the embedding attribution calculation.

### Student extension

Try changing the reference construction in Cell 15 while preserving:
- The input tensor shape.
- The `[CLS]` token position.
- The `[SEP]` token position.
- The attention mask shape.

Then rerun Cells 17–24 and compare the token scores. Do not change the model or target class during this comparison.

---

## 21. Optional: analyse a custom sentence

Students can replace the selected sentence with their own short review. They must rerun the tokenization, prediction, baseline, target-class selection, and attribution cells.

### Cell 28 – Set a custom review

```python
custom_text = (
    "The product looks beautiful and works well, "
    "although the delivery was very slow."
)

selected_text = custom_text

encoded = tokenizer(
    selected_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH,
    padding="max_length"
)

input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

with torch.no_grad():
    custom_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).logits

    custom_probabilities = torch.softmax(custom_logits, dim=-1)

predicted_class_id = int(torch.argmax(custom_logits, dim=-1).item())
TARGET_CLASS_ID = predicted_class_id

print("Custom sentence:", selected_text)
print("Predicted sentiment:", label_names[predicted_class_id])
print("Probabilities:", custom_probabilities.detach().cpu().numpy())
```

After running this cell, rerun:
- Cell 11 only if you want to inspect the new tokens.
- Cell 15 to rebuild the baseline.
- Cell 16 to update the target forward function.
- Cells 17–24 to calculate and display the new attribution.

For the opposite-class experiment, rerun Cells 25–27 as well.

---

## 22. Troubleshooting

### Issue 1 – Hugging Face download fails

**Possible cause:** The Colab runtime cannot connect to Hugging Face or the service is temporarily unavailable.

**Try:**
- Confirm that the runtime has internet access.
- Rerun the dataset/model loading cell.
- Check the model and dataset IDs for typing errors.
- If the Hub is temporarily unavailable, retry later.

### Issue 2 – Model does not have `distilbert.embeddings`

Inspect the model type:

```python
print(type(model))
print(model.base_model_prefix)
print(model)
```

This lab expects `AutoModelForSequenceClassification` to load the specified DistilBERT checkpoint. The embedding layer path for this checkpoint is `model.distilbert.embeddings`.

### Issue 3 – Captum reports a gradient or integer-input error

Do not use ordinary `IntegratedGradients` directly on integer token IDs. Token IDs are discrete indices. Use `LayerIntegratedGradients` with the embedding layer, as demonstrated in this lab.

### Issue 4 – Attribution shape is unexpected

Inspect the shape:

```python
print("Input shape:", input_ids.shape)
print("Attribution shape:", attributions.shape)
```

For this lab's single-sentence input, the attribution tensor should correspond to batch size 1, sequence length, and the embedding dimension of the selected layer.

### Issue 5 – GPU out-of-memory error

Use a CPU runtime, or reduce:
- `MAX_LENGTH` from `64` to `48`.
- `n_steps` from `50` to `25`.

The selected sentence must still fit within the chosen maximum length.

### Issue 6 – No visible attribution for some tokens

Some tokens may receive scores close to zero. This is valid. It means the token has a small attribution under the selected target class, model, and baseline. It does not mean the token is universally unimportant.

### Issue 7 – WordPiece tokens look fragmented

This is expected. DistilBERT uses WordPiece tokenization. Tokens beginning with `##` are continuation fragments. The coloured-token cell removes `##` for display only; the attribution calculation still operates on the original model tokens.

---

## 23. Key concepts to reinforce

| Concept | Explanation |
|---|---|
| Transformer | Neural network architecture that uses attention mechanisms to process token sequences |
| DistilBERT | A smaller distilled Transformer based on BERT |
| Tokenizer | Converts text into model-specific token IDs |
| WordPiece | Subword tokenisation used by BERT-family models |
| Token embedding | Continuous vector representation of a token |
| Attention mask | Indicates which positions are actual input tokens and which positions are padding |
| Logit | Raw model score before softmax |
| Layer Integrated Gradients | Attributes a model output to the inputs or outputs of a selected layer |
| Baseline/reference | The comparison input from which the attribution path begins |
| Positive attribution | Pushes the selected target score upward relative to the baseline |
| Negative attribution | Pushes the selected target score downward relative to the baseline |
| Completeness | Sum of attributions approximates the difference between model output at the input and at the baseline |
| Convergence delta | Captum's approximation diagnostic for the completeness property |

### Token attribution vs. attention weights

| Attention weights | Token attribution |
|---|---|
| Describe how attention heads distribute attention across positions | Attribute a selected output score to input tokens |
| Are internal Transformer attention quantities | Are calculated using an attribution method such as Layer Integrated Gradients |
| Do not automatically explain why a class was predicted | Are tied to a specific output class and reference input |
| Are not interchangeable with feature importance | Still require careful interpretation and are not causal proof |

---

## 24. Student activities

### Activity 1 – Explain the prediction

For the selected sentence, write down:
- The original sentence.
- The model's predicted sentiment.
- The negative and positive probabilities.
- The three tokens with the largest positive attribution for the predicted class.
- The three tokens with the largest negative attribution for the predicted class.

### Activity 2 – Compare both classes

Calculate attributions for both sentiment classes.

Answer:
- Which tokens support the predicted class?
- Which tokens push against it?
- Are the patterns identical or different?
- Why does the explanation need to identify the target class?

### Activity 3 – Try a mixed-sentiment sentence

Use a sentence that has both positive and negative phrases, for example:

```python
selected_text = (
    "The acting was excellent, but the story was slow and predictable."
)
```

Run the prediction and attribution again.

Discuss whether the model seems to focus on one phrase more than another. Does the overall sentiment prediction reflect the whole sentence?

### Activity 4 – Change one word

Make a small edit to the sentence. For example, change `excellent` to `terrible`, or `disappointing` to `wonderful`.

Re-run the prediction and attribution. Observe:
- Did the predicted sentiment change?
- Did the probability change?
- Did the most influential tokens change?
- Did the tokenisation change?

### Activity 5 – Examine WordPiece fragmentation

Choose a sentence containing a longer or less common word. Inspect how the tokenizer splits it.

Discuss:
- Why does the model use subword tokens?
- Should token attributions be aggregated into complete words for a user-facing explanation?
- What information may be lost when multiple subword scores are combined?

---

## 25. Knowledge-check questions

1. Why does a Transformer need a tokenizer?
2. What is a WordPiece token?
3. What does `[CLS]` represent in this classification model?
4. Why can we not calculate ordinary gradients directly with respect to integer token IDs?
5. What is the purpose of `LayerIntegratedGradients`?
6. Which layer is selected for this DistilBERT model?
7. What is the reference input used in this lab?
8. What does a positive token attribution mean when the target class is `NEGATIVE`?
9. What does the completeness property check?
10. Are attention weights and token attributions the same?
11. Does a high token attribution prove that the word caused the model to predict a sentiment?
12. Why should the target class and baseline be stated whenever token attributions are presented?

### Suggested answers

1. It converts text into token IDs that the model can process.
2. It is a subword unit; one word can be represented by several WordPiece tokens.
3. It is a special classification token used by the model's classification head.
4. Token IDs are discrete integer indices, not continuous variables with meaningful gradients.
5. It attributes a model output to the inputs or outputs of a selected internal layer.
6. `model.distilbert.embeddings`.
7. A sequence that keeps `[CLS]` and `[SEP]` but replaces ordinary tokens with `[PAD]`.
8. It pushes the model's negative-class logit upward relative to the baseline.
9. It checks whether the sum of attributions approximates the output difference between the actual and reference inputs.
10. No. Attention weights describe attention distributions; token attribution explains a selected output using an attribution method.
11. No. It describes the model's score relative to a baseline, not a causal effect.
12. Because attribution values depend on the selected output and comparison input.

---

## 26. Suggested 60–75 minute instructor flow

| Time | Demonstration |
|---|---|
| 0–5 min | Introduce the sentiment classification case study |
| 5–10 min | Load SST-2 and inspect labels |
| 10–20 min | Load DistilBERT and run predictions |
| 20–30 min | Explain tokenisation, WordPiece, embeddings, and attention masks |
| 30–40 min | Explain the reference input and Layer Integrated Gradients |
| 40–50 min | Calculate attributions and show the token score table |
| 50–60 min | Display coloured tokens and verify completeness |
| 60–70 min | Compare positive/negative target classes and run a custom example |
| 70–75 min | Discuss limitations and conduct the knowledge check |

For a 60-minute session, use one selected sentence and treat the custom-sentence activity as an extension.

---

## 27. Responsible interpretation and limitations

Emphasise these points:

1. **Attribution is not causation.** A token with a high attribution is not proven to cause the model's prediction.
2. **The baseline matters.** A different reference input can produce different scores.
3. **The target class matters.** Attributions for positive and negative logits answer different questions.
4. **Tokenisation matters.** A word may be split into multiple subword tokens.
5. **Context matters.** The same word can influence different sentences differently.
6. **Model bias matters.** A pretrained sentiment model may reflect biases in its training data. Its prediction should not be treated as an objective judgement of a person or product.
7. **Attribution is model-specific.** Another Transformer, checkpoint, or layer may produce different scores.
8. **A correct-looking explanation is not proof of model quality.** Evaluate model performance, robustness, and fairness separately.

---

## 28. Closing explanation for students

Use this summary to conclude the demonstration:

> “A Transformer does not read a sentence exactly the way a human does. It first converts the sentence into tokens and then into embedding vectors. Layer Integrated Gradients helps us estimate how each token's embedding contributes to a selected model output relative to a reference input. This lets us highlight words that push the prediction up or down. But token attribution is not the same as attention, and it does not prove that a word caused the model's decision. We must always state the target class, baseline, and limitations.”

### Connection to the previous labs

- **Lab 3 – DeepSHAP / DeepLIFT:** Attributed tabular input features in a neural network.
- **Lab 4 – Counterfactual Explanations:** Searched for alternative tabular inputs that change a prediction.
- **Lab 5 – Integrated Gradients on LSTM:** Attributed a time-series forecast to input time steps.
- **Lab 6 – Transformer Token Attribution:** Attributes a sentiment prediction to input text tokens.

Across all four labs, the central question is the same:

**“What aspects of the input influenced the model's output, and what are the limitations of that explanation?”**

---

## References

- Hugging Face model card – DistilBERT fine-tuned on SST-2:  
  [https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- Hugging Face GLUE dataset – SST-2:  
  [https://huggingface.co/datasets/nyu-mll/glue/tree/main/sst2](https://huggingface.co/datasets/nyu-mll/glue/tree/main/sst2)
- Captum – Integrated Gradients and NLP FAQ:  
  [https://captum.ai/docs/faq](https://captum.ai/docs/faq)
- Captum – BERT model interpretability tutorial:  
  [https://captum.ai/tutorials/Bert_SQUAD_Interpret](https://captum.ai/tutorials/Bert_SQUAD_Interpret)
- Captum – Layer Integrated Gradients API:  
  [https://captum.ai/api/layer.html](https://captum.ai/api/layer.html)
- Hugging Face Transformers documentation:  
  [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

**End of Lab 6**
