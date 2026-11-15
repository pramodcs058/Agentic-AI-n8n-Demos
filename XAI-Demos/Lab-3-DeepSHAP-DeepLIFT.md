# Lab 3 -- DeepSHAP and DeepLIFT: Explaining Tabular Neural-Network Predictions

**Platform:** Google Colab\
**Runtime:** CPU (GPU optional; not required)\
**Level:** Intermediate\
**Estimated time:** 25--30 minutes\
**Dataset:** Breast Cancer Wisconsin Diagnostic dataset (built into
scikit-learn)\
**Model:** Small feed-forward neural network built with PyTorch\
**Libraries:** PyTorch, scikit-learn, SHAP, Captum, NumPy, pandas,
Matplotlib

------------------------------------------------------------------------

## 1. Lab objective

In this lab, you will explain a neural network's prediction using two
feature-attribution techniques:

-   **DeepSHAP:** An approximation of SHAP values for deep learning
    models. It uses a background dataset and DeepLIFT-style propagation
    rules to estimate each feature's contribution.
-   **DeepLIFT:** Deep Learning Important FeaTures. It compares a real
    input with a reference (baseline) input and propagates contribution
    scores through the network.

Unlike Occlusion and Grad-CAM, which explain image regions, this lab
uses tabular data. Each input column is a measurable feature, so the
explanation can be displayed as a ranked bar chart.

The lab follows the slide 33 plan: use a small dataset, train a small
neural network, apply SHAP and Captum, and display positive and negative
feature contributions.

> Dataset choice: The Breast Cancer Wisconsin Diagnostic dataset is
> included with scikit-learn, so no external CSV download is needed. It
> is small, reproducible and suitable for a classroom demonstration. The
> task is educational; this model is not a medical diagnostic system.

------------------------------------------------------------------------

## 2. Learning outcomes

By the end of this lab, students should be able to:

-   Load and explore a real tabular classification dataset.
-   Train a small feed-forward neural network using PyTorch.
-   Explain why input features must be scaled.
-   Select a background/reference dataset for attribution.
-   Generate DeepSHAP feature contributions.
-   Generate DeepLIFT feature contributions using Captum.
-   Interpret positive and negative contribution scores.
-   Compare the assumptions and limitations of the two methods.

------------------------------------------------------------------------

## 3. Classroom scenario

Imagine a clinical research team has trained a machine-learning model to
classify breast tumour measurements as either:

-   **Malignant** -- the target class we will treat as the positive
    class (`1`).
-   **Benign** -- the negative class (`0`).

A researcher asks:

> "The model has predicted a high probability of malignant. Which
> measurements contributed to that prediction?"

The model's output alone is not enough. We want to investigate how the
input features contributed to the output for one particular sample.

### Opening questions

Ask students:

1.  If the model predicts malignant, does every feature increase the
    malignant probability?
2.  Could some features push the prediction towards malignant while
    others push it towards benign?
3.  What should we compare the applicant/patient data against to
    calculate a contribution?
4.  Does a feature with a large SHAP value necessarily cause the
    outcome?

Return to these questions after the plots are generated.

------------------------------------------------------------------------

## 4. Start Google Colab

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Select **New notebook**.
3.  Rename it to `Lab-3-DeepSHAP-DeepLIFT`.
4.  Select **Runtime → Change runtime type → CPU**.
5.  Execute the cells in order.

The dataset is bundled with scikit-learn. No dataset download or paid
API is required. SHAP may download/install its package dependencies
during the first setup.

------------------------------------------------------------------------

## 5. Step 1 -- Install and import libraries

Install compatible, pinned versions of SHAP and Captum. Pinning helps
reduce package-version surprises during a live demonstration.

``` python
%pip -q install shap==0.46.0 captum==0.7.0
```

Import the libraries.

``` python
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import shap
from captum.attr import DeepLift

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

Set seeds and select the runtime device.

``` python
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch:", torch.__version__)
print("SHAP:", shap.__version__)
print("Device:", device)
```

**Instructor explanation**

-   PyTorch builds and trains the neural network.
-   SHAP provides the `DeepExplainer` implementation.
-   Captum provides `DeepLift`.
-   scikit-learn supplies the dataset, train/test split, feature scaling
    and evaluation metrics.
-   We set random seeds to make the training run more repeatable.

If Colab reports that a package was already imported before
installation, restart the runtime after the installation cell and rerun
the notebook from the top.

------------------------------------------------------------------------

## 6. Step 2 -- Load the dataset

The Breast Cancer Wisconsin Diagnostic dataset contains 569 samples and
30 numeric features derived from digitised images of breast masses.

The original scikit-learn target encoding is:

-   `0` = malignant
-   `1` = benign

For this lab, we deliberately define **malignant as class 1**, making
the positive class easy to discuss.

``` python
dataset = load_breast_cancer()

X = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

# Convert labels so that 1 = malignant and 0 = benign.
y = (dataset.target == 0).astype(np.float32)

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("\nClass counts:")
print(pd.Series(y).map({0.0: "Benign", 1.0: "Malignant"}).value_counts())
```

Display the first few rows.

``` python
display(X.head())
```

Display the feature names.

``` python
for index, feature in enumerate(X.columns, start=1):
    print(f"{index:02d}. {feature}")
```

**Instructor explanation**

Each row represents one sample. Each column represents a measured
feature, such as mean radius, mean texture or worst perimeter.

The feature names include terms such as:

-   `mean` -- average measurement.
-   `se` -- standard error.
-   `worst` -- the largest or most extreme measurement among the
    computed values.

For example, `mean radius` is one input feature. The neural network
receives all 30 feature values together to make a prediction.

**Important:** The labels are used for supervised training. They are not
included as input features.

------------------------------------------------------------------------

## 7. Step 3 -- Explore the target distribution

Plot the number of benign and malignant samples.

``` python
class_counts = pd.Series(y).map({
    0.0: "Benign",
    1.0: "Malignant"
}).value_counts()

plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar")
plt.title("Breast Cancer Dataset – Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.xticks(rotation=0)
plt.show()
```

**Ask students**

-   Is this a binary classification problem?
-   What does class `1` represent in our notebook?
-   Why should we be careful when interpreting model predictions in a
    medical context?

------------------------------------------------------------------------

## 8. Step 4 -- Split the data into training and testing sets

We must split the data before fitting the scaler. This prevents
information from the test set leaking into preprocessing.

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
```

**Instructor explanation**

-   The training set is used to learn model parameters.
-   The test set is held back to evaluate performance on unseen samples.
-   `stratify=y` preserves approximately the same class proportions in
    both sets.

------------------------------------------------------------------------

## 9. Step 5 -- Standardise the features

The 30 input features have different numeric ranges. For example, area
measurements can be much larger than smoothness measurements.

Standardisation transforms each feature using the training-set mean and
standard deviation:

\[ z = `\frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}`{=tex} \]

Fit the scaler only on the training data, then transform both sets.

``` python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)
```

Convert the data into PyTorch tensors.

``` python
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.float32
).view(-1, 1)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).view(-1, 1)

print("X training tensor:", X_train_tensor.shape)
print("y training tensor:", y_train_tensor.shape)
```

**Instructor explanation**

The model is trained on standardised values. The explanations will
therefore refer to the same scaled input representation.

The baseline used later must also be expressed in this scaled feature
space.

------------------------------------------------------------------------

## 10. Step 6 -- Build a small neural network

We will use a simple feed-forward neural network:

-   Input layer: 30 features.
-   Hidden layer 1: 32 neurons with ReLU.
-   Hidden layer 2: 16 neurons with ReLU.
-   Output layer: 1 neuron with Sigmoid.

``` python
class BreastCancerMLP(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


model = BreastCancerMLP(
    input_features=X_train_tensor.shape[1]
).to(device)

print(model)
```

**Instructor explanation**

-   The input layer receives all 30 measurements.
-   Hidden layers learn combinations and non-linear patterns.
-   ReLU introduces non-linearity.
-   Sigmoid converts the final output into a probability between 0 and
    1.

In this notebook:

-   Probability closer to 1 means the model predicts malignant.
-   Probability closer to 0 means the model predicts benign.

------------------------------------------------------------------------

## 11. Step 7 -- Train the neural network

Create a DataLoader.

``` python
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

Define the loss function and optimiser.

``` python
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Train for 100 epochs.

``` python
EPOCHS = 100

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_dataset)
    loss_history.append(epoch_loss)

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch + 1:3d}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f}"
        )
```

Plot the training loss.

``` python
plt.figure(figsize=(7, 4))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.grid(True)
plt.show()
```

**Instructor explanation**

The model learns by:

1.  Making a prediction.
2.  Comparing the prediction with the known label.
3.  Calculating the loss.
4.  Backpropagating gradients.
5.  Updating weights using Adam.

We are training a deliberately small neural network so that students can
focus on explainability rather than model complexity.

------------------------------------------------------------------------

## 12. Step 8 -- Evaluate the model

Evaluate on the held-out test set.

``` python
model.eval()

with torch.no_grad():
    test_probabilities = model(
        X_test_tensor.to(device)
    ).cpu().numpy().ravel()

test_predictions = (test_probabilities >= 0.5).astype(int)

print(
    "Test accuracy:",
    f"{accuracy_score(y_test, test_predictions):.3f}"
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Benign", "Malignant"]
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))
```

**Instructor explanation**

This evaluation gives a basic indication of how the trained model
performs on held-out samples. It does not establish clinical
suitability.

For this demonstration, our next task is not to improve the model. It is
to explain one of its predictions.

------------------------------------------------------------------------

## 13. Step 9 -- Select one sample to explain

Select a test sample that the model predicts as malignant with a
relatively high probability. This makes the example easier to discuss.

``` python
# Select the test sample with the highest predicted malignant probability.
sample_index = int(np.argmax(test_probabilities))

sample = X_test_tensor[sample_index:sample_index + 1].to(device)
sample_probability = float(test_probabilities[sample_index])
actual_label = int(y_test[sample_index])

print("Selected test row:", sample_index)
print("Actual class:", "Malignant" if actual_label == 1 else "Benign")
print(f"Predicted malignant probability: {sample_probability:.4f}")
print(
    "Model prediction:",
    "Malignant" if sample_probability >= 0.5 else "Benign"
)
```

Create a readable table for this sample using the original feature
values.

``` python
sample_original = X_test.iloc[[sample_index]].T
sample_original.columns = ["Original measurement"]

display(sample_original)
```

**Instructor explanation**

We have selected one prediction. This is a **local explanation**: we
want to explain this particular sample, not the behaviour of the whole
model.

For teaching, selecting a high-confidence malignant prediction is
convenient. It does not mean the prediction is medically correct.

------------------------------------------------------------------------

## 14. Step 10 -- Understand the baseline/background data

Both methods compare the selected sample with a reference:

-   DeepSHAP uses a **background dataset** to estimate the expected
    model output and feature contributions.
-   DeepLIFT compares the input with a **reference or baseline input**.

For this lab, select 50 training samples as the SHAP background dataset.
Use the mean of those same background samples as the DeepLIFT reference.

``` python
BACKGROUND_SIZE = 50

background = X_train_tensor[:BACKGROUND_SIZE].to(device)

# One reference row: mean of the selected background data.
baseline = background.mean(dim=0, keepdim=True)

print("SHAP background shape:", background.shape)
print("DeepLIFT baseline shape:", baseline.shape)
```

**Instructor explanation**

The background data represents a reference population for SHAP. It helps
define what the model would typically output in the absence of the
particular sample's feature values.

The DeepLIFT baseline is a reference input. Because the data was
standardised, a vector of zeros corresponds approximately to the
training-set feature means. Here we explicitly calculate the mean of the
selected background rows.

**Important:** Feature attribution depends on the chosen background or
baseline. Changing it can change the explanation.

------------------------------------------------------------------------

# Part A -- DeepSHAP using SHAP

## 15. Step 11 -- Create a DeepSHAP explainer

SHAP provides `DeepExplainer` for supported deep-learning models.

``` python
model.eval()

deep_shap_explainer = shap.DeepExplainer(
    model,
    background
)

print("DeepSHAP explainer created.")
```

**Instructor explanation**

The explainer receives:

1.  The trained neural network.
2.  A background dataset.

DeepSHAP uses the background data and DeepLIFT-related propagation rules
to approximate Shapley feature contributions for the model.

It does not retrain the network.

------------------------------------------------------------------------

## 16. Step 12 -- Calculate DeepSHAP values

Calculate attributions for the selected sample.

``` python
shap_values_raw = deep_shap_explainer.shap_values(sample)

print("Raw SHAP output type:", type(shap_values_raw))
```

SHAP output shapes can differ between versions and model-output formats.
The following cell converts the result into one attribution value per
feature for this single-output model.

``` python
# DeepExplainer may return a list for multi-output models.
if isinstance(shap_values_raw, list):
    shap_values_array = np.asarray(shap_values_raw[0])
else:
    shap_values_array = np.asarray(shap_values_raw)

# Remove the single-sample dimension.
shap_values_array = np.squeeze(shap_values_array)

# For this one-output model, remove a trailing singleton output dimension
# if the installed SHAP version returns one.
if shap_values_array.ndim == 2 and shap_values_array.shape[-1] == 1:
    shap_values_array = shap_values_array[:, 0]

shap_values_one = shap_values_array.reshape(-1)

if shap_values_one.shape[0] != X_train.shape[1]:
    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{shap_values_array.shape}. "
        "Expected one attribution per input feature."
    )

print("Number of SHAP feature values:", len(shap_values_one))
```

**Instructor explanation**

There should be one SHAP value for each of the 30 input features.

-   Positive SHAP value: pushes the model output above its
    expected/reference output.
-   Negative SHAP value: pushes the model output below its
    expected/reference output.
-   Value close to zero: has little contribution relative to the chosen
    background for this sample.

Because our model outputs malignant probability, positive contributions
push that probability higher and negative contributions push it lower.

------------------------------------------------------------------------

## 17. Step 13 -- Read the DeepSHAP expected value

``` python
expected_value_raw = deep_shap_explainer.expected_value

expected_value = float(
    np.asarray(expected_value_raw).reshape(-1)[0]
)

print(f"DeepSHAP expected output: {expected_value:.4f}")
print(f"Selected sample output:  {sample_probability:.4f}")
print(f"Sum of SHAP values:      {shap_values_one.sum():.4f}")
print(
    "Expected output + SHAP sum:",
    f"{expected_value + shap_values_one.sum():.4f}"
)
```

The additive explanation is approximately:

\[ f(x) `\approx `{=tex}E\[f(X)\] + `\sum`{=tex}\_i `\phi`{=tex}\_i \]

Where:

-   (f(x)) is the model output for the selected sample.
-   (E\[f(X)\]) is the expected model output over the background data.
-   (`\phi`{=tex}\_i) is the contribution assigned to feature (i).

Small numerical differences can occur because DeepSHAP is an
approximation and because of numerical precision.

**Teaching point:** SHAP values explain how features move the model
output away from a reference expectation. They do not prove that the
features cause the real-world outcome.

------------------------------------------------------------------------

## 18. Step 14 -- Visualise DeepSHAP feature contributions

Create a horizontal bar chart showing the largest positive and negative
contributions.

``` python
shap_series = pd.Series(
    shap_values_one,
    index=X.columns
)

top_features = shap_series.reindex(
    shap_series.abs().sort_values(ascending=False).head(12).index
).sort_values()

plt.figure(figsize=(10, 7))

bar_colors = [
    "tomato" if value > 0 else "steelblue"
    for value in top_features.values
]

plt.barh(
    top_features.index,
    top_features.values,
    color=bar_colors
)

plt.axvline(0, color="black", linewidth=0.8)
plt.title("DeepSHAP – Feature Contributions")
plt.xlabel("SHAP value (change in malignant probability)")
plt.ylabel("Input feature")
plt.tight_layout()
plt.show()
```

**How to read the chart**

-   Red/right/positive bars push the malignant probability upward.
-   Blue/left/negative bars push the malignant probability downward.
-   Longer bars represent larger contribution magnitudes for this
    sample.

The chart shows the 12 features with the largest absolute SHAP values.

**Ask students**

1.  Which three features have the largest absolute contributions?
2.  Which features push the model towards malignant?
3.  Which features push the model towards benign?
4.  Would the same features necessarily be important for another
    patient/sample?

------------------------------------------------------------------------

# Part B -- DeepLIFT using Captum

## 19. Step 15 -- Create a DeepLIFT explainer

Captum provides `DeepLift` for PyTorch models.

``` python
model.eval()

deeplift = DeepLift(model)

print("DeepLIFT explainer created.")
```

**Instructor explanation**

DeepLIFT compares the activation of each neuron for the selected input
against its activation for a reference input.

Instead of relying only on local gradients, it propagates contribution
scores based on differences from the reference.

The reference input is important. A different baseline can lead to
different contribution values.

------------------------------------------------------------------------

## 20. Step 16 -- Calculate DeepLIFT attributions

Use the same selected sample and the mean background baseline.

``` python
sample_for_deeplift = sample.detach().clone().requires_grad_(True)

deeplift_attributions, convergence_delta = deeplift.attribute(
    inputs=sample_for_deeplift,
    baselines=baseline,
    return_convergence_delta=True
)

deeplift_values = (
    deeplift_attributions
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
)

print("Number of DeepLIFT feature values:", len(deeplift_values))
print("Convergence delta:", convergence_delta.detach().cpu().numpy())
```

**Instructor explanation**

-   `inputs` is the sample being explained.
-   `baselines` is the reference feature vector.
-   `attribute()` calculates a contribution for every input feature.
-   `convergence_delta` helps assess how closely the attributions
    account for the difference between the model output at the input and
    at the baseline.

The output contains 30 attribution values, one per feature.

------------------------------------------------------------------------

## 21. Step 17 -- Check the DeepLIFT completeness relationship

For this single-output probability model, the sum of the DeepLIFT
attributions should approximately equal:

\[ f(x) - f(x\_{`\text{baseline}`{=tex}}) \]

Calculate the values.

``` python
with torch.no_grad():
    input_output = model(sample).item()
    baseline_output = model(baseline).item()

attribution_sum = float(deeplift_values.sum())
output_difference = input_output - baseline_output

print(f"Model output for sample:       {input_output:.6f}")
print(f"Model output for baseline:     {baseline_output:.6f}")
print(f"Output difference:             {output_difference:.6f}")
print(f"Sum of DeepLIFT attributions:  {attribution_sum:.6f}")
print(f"Captum convergence delta:      {convergence_delta.item():.6f}")
```

**Instructor explanation**

DeepLIFT is designed to satisfy a summation-to-delta property:
contributions add up to the difference between the model's output for
the input and the baseline, subject to numerical and implementation
effects.

A small convergence delta indicates that the attribution sum is close to
the output difference. It is a numerical consistency check, not proof
that the explanation is correct or causal.

------------------------------------------------------------------------

## 22. Step 18 -- Visualise DeepLIFT feature contributions

Create a horizontal bar chart for the 12 features with the largest
absolute contributions.

``` python
deeplift_series = pd.Series(
    deeplift_values,
    index=X.columns
)

top_deeplift = deeplift_series.reindex(
    deeplift_series.abs().sort_values(ascending=False).head(12).index
).sort_values()

plt.figure(figsize=(10, 7))

bar_colors = [
    "tomato" if value > 0 else "steelblue"
    for value in top_deeplift.values
]

plt.barh(
    top_deeplift.index,
    top_deeplift.values,
    color=bar_colors
)

plt.axvline(0, color="black", linewidth=0.8)
plt.title("DeepLIFT – Feature Contributions")
plt.xlabel("Contribution to malignant probability")
plt.ylabel("Input feature")
plt.tight_layout()
plt.show()
```

**How to read the chart**

-   Positive contributions push the model output above its baseline
    output.
-   Negative contributions push the model output below its baseline
    output.
-   Larger absolute values indicate larger contributions relative to the
    selected reference.

**Ask students**

-   Are the same features important in DeepSHAP and DeepLIFT?
-   Are the signs of their contributions the same?
-   What could explain differences between the two methods?

------------------------------------------------------------------------

## 23. Step 19 -- Compare DeepSHAP and DeepLIFT

The two techniques are related but not identical.

  -----------------------------------------------------------------------
  Aspect                  DeepSHAP                DeepLIFT
  ----------------------- ----------------------- -----------------------
  Main idea               Approximate Shapley     Propagate contributions
                          feature contributions   based on differences
                          for deep models         from a reference

  Reference               Background dataset      Baseline input

  Library used            SHAP                    Captum

  Output                  Feature attribution     Feature contribution
                          values                  values

  Positive value          Raises output relative  Raises output relative
                          to expected output      to baseline output

  Negative value          Lowers output relative  Lowers output relative
                          to expected output      to baseline output

  Key dependency          Background data and     Baseline choice and
                          SHAP approximation      propagation rules

  Typical use             Feature attribution for Efficient
                          deep models             reference-based
                                                  attribution
  -----------------------------------------------------------------------

### Display both methods side by side

``` python
comparison = pd.DataFrame({
    "DeepSHAP": shap_values_one,
    "DeepLIFT": deeplift_values
}, index=X.columns)

# Select features with the largest average absolute attribution.
comparison["average_absolute"] = (
    comparison[["DeepSHAP", "DeepLIFT"]].abs().mean(axis=1)
)

selected_features = (
    comparison["average_absolute"]
    .sort_values(ascending=False)
    .head(12)
    .index
)

comparison_plot = comparison.loc[selected_features, [
    "DeepSHAP", "DeepLIFT"
]]

comparison_plot = comparison_plot.loc[
    comparison_plot.abs().max(axis=1).sort_values().index
]

comparison_plot.plot(
    kind="barh",
    figsize=(11, 8)
)

plt.axvline(0, color="black", linewidth=0.8)
plt.title("DeepSHAP vs DeepLIFT – Feature Contributions")
plt.xlabel("Contribution to malignant probability")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
```

**Important interpretation note**

The two methods use different reference constructions and attribution
rules. Compare the direction and relative patterns thoughtfully; do not
assume their numeric values must match exactly.

------------------------------------------------------------------------

## 24. Step 20 -- Student experiments

### Activity A -- Explain another sample

Choose a different test sample.

``` python
another_index = 0

another_sample = X_test_tensor[
    another_index:another_index + 1
].to(device)

with torch.no_grad():
    another_probability = model(another_sample).item()

print("New sample malignant probability:", another_probability)
```

Calculate DeepSHAP and DeepLIFT for this sample. Compare its feature
contributions with the first sample.

**Discussion:** Does the same feature always contribute in the same
direction?

### Activity B -- Change the DeepLIFT baseline

Compare the mean baseline with a zero baseline.

``` python
zero_baseline = torch.zeros_like(baseline)

zero_attributions, zero_delta = deeplift.attribute(
    inputs=sample_for_deeplift,
    baselines=zero_baseline,
    return_convergence_delta=True
)

zero_values = zero_attributions.detach().cpu().numpy().reshape(-1)

baseline_comparison = pd.DataFrame({
    "Mean baseline": deeplift_values,
    "Zero baseline": zero_values
}, index=X.columns)

baseline_comparison.head(10)
```

**Discussion**

-   Did the feature contributions change?
-   Why is the baseline choice important?
-   In standardised data, what does a zero input approximately
    represent?

### Activity C -- Change the SHAP background size

Try a different number of training examples for the background dataset.

``` python
small_background = X_train_tensor[:20].to(device)

small_explainer = shap.DeepExplainer(
    model,
    small_background
)

small_shap_raw = small_explainer.shap_values(sample)

if isinstance(small_shap_raw, list):
    small_shap_array = np.asarray(small_shap_raw[0])
else:
    small_shap_array = np.asarray(small_shap_raw)

small_shap_array = np.squeeze(small_shap_array)

if small_shap_array.ndim == 2 and small_shap_array.shape[-1] == 1:
    small_shap_array = small_shap_array[:, 0]

small_shap_values = small_shap_array.reshape(-1)

print("Original background size:", len(background))
print("New background size:", len(small_background))
print(
    "Mean absolute difference in SHAP values:",
    np.mean(np.abs(shap_values_one - small_shap_values))
)
```

**Discussion:** Why can the background dataset affect DeepSHAP
explanations?

------------------------------------------------------------------------

## 25. Troubleshooting

  -----------------------------------------------------------------------
  Problem                 Possible reason         Solution
  ----------------------- ----------------------- -----------------------
  SHAP import error after Package was imported    Restart Colab runtime
  installation            before upgrade/install  and rerun from the
                                                  first cell

  SHAP version conflict   Existing NumPy or       Keep the pinned
                          package dependency      versions; restart
                          mismatch                runtime after
                                                  installation

  DeepExplainer           Unsupported model       Keep the small
  unsupported operation   operation or layer      Linear/ReLU/Sigmoid
                                                  network used in this
                                                  lab

  SHAP output shape       SHAP version or model   Use the
  differs                 output shape            shape-normalisation
                                                  cell and inspect the
                                                  raw output

  Captum attribution      Unsupported layer or    Use the provided
  error                   in-place operation      Linear/ReLU network;
                                                  avoid adding in-place
                                                  activations

  Convergence delta not   Baseline or numerical   Check that input and
  close to zero           effects                 baseline have identical
                                                  feature order and shape

  All contributions look  Sample is close to      Inspect the output
  tiny                    baseline or model       difference and
                          output is insensitive   attribution sum

  Model predicts only one Training or threshold   Inspect class
  class                   issue                   distribution, training
                                                  loss and probability
                                                  outputs
  -----------------------------------------------------------------------

### If SHAP raises a compatibility error

1.  Run the installation cell.
2.  Select **Runtime → Restart session**.
3.  Run all cells again from the top.
4.  Confirm the printed SHAP version is `0.46.0`.

Do not install multiple SHAP versions during the live class without
restarting the runtime.

------------------------------------------------------------------------

## 26. Limitations and responsible interpretation

Discuss these points with students:

1.  **Reference dependency:** DeepSHAP depends on the selected
    background dataset; DeepLIFT depends on its baseline.
2.  **Feature dependence:** Correlated input features can make
    attribution difficult to interpret. The breast-cancer measurements
    include related size and shape features.
3.  **Approximation:** DeepSHAP is an approximation of Shapley-value
    explanations for deep models.
4.  **Model explanation, not causation:** Attribution indicates how the
    model's output is associated with input features under the
    explanation method. It does not establish that changing a feature in
    the real world will cause the outcome to change.
5.  **One prediction versus the whole model:** The charts explain one
    selected sample. They do not automatically describe global model
    behaviour.
6.  **Dataset limitations:** The dataset is a teaching dataset. It is
    not a substitute for clinical validation or professional medical
    judgement.

**Important classroom message:** A feature with a large SHAP or DeepLIFT
contribution is not automatically a causal risk factor.

------------------------------------------------------------------------

## 27. Final discussion and knowledge check

Ask students to answer without referring to the notebook.

1.  Why do we use a background dataset in DeepSHAP?
2.  What is a baseline in DeepLIFT?
3.  What does a positive SHAP value mean in this lab?
4.  What does a negative DeepLIFT value mean?
5.  Why do we standardise the features?
6.  Why must the baseline use the same feature order as the input?
7.  Why might DeepSHAP and DeepLIFT produce different values?
8.  Does a large attribution prove causation?
9.  Is the explanation local or global?
10. What does the DeepLIFT convergence delta check?

### Suggested answers

  -----------------------------------------------------------------------
  Question                            Expected answer
  ----------------------------------- -----------------------------------
  DeepSHAP background?                It provides reference samples for
                                      expected output and attribution
                                      estimation

  DeepLIFT baseline?                  A reference input against which the
                                      actual input is compared

  Positive SHAP?                      Pushes malignant probability above
                                      the expected output

  Negative DeepLIFT?                  Pushes the output below the
                                      baseline output

  Why standardise?                    Puts features on comparable numeric
                                      scales and supports neural-network
                                      training

  Same feature order?                 Each attribution must correspond to
                                      the correct input feature

  Why different values?               Different reference data and
                                      attribution rules

  Does attribution prove causation?   No

  Local or global?                    Local: one selected sample

  Convergence delta?                  Checks how closely attribution sums
                                      match the output difference
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 28. Lab conclusion

In this lab, we trained a small feed-forward neural network using the
Breast Cancer Wisconsin Diagnostic dataset. We selected one test sample
and explained its predicted malignant probability using DeepSHAP and
DeepLIFT.

We visualised positive and negative feature contributions and compared
how the two reference-based techniques explained the same prediction.

### Key takeaway

**A neural network gives us a prediction. DeepSHAP and DeepLIFT help us
investigate how individual input features contributed to that prediction
relative to a chosen reference.**

### Link to the next lab -- Counterfactual Explanations

Feature attribution tells us which features contributed to the current
prediction. But a user may ask a different question:

> "What is the smallest realistic change to my input that would make the
> model produce a different prediction?"

That introduces counterfactual explanations.

In Lab 4, we will use a loan-approval scenario and generate alternative
applicant profiles to investigate how a prediction can change.

------------------------------------------------------------------------

## References

-   [Google Colab](https://colab.research.google.com/)
-   [scikit-learn Breast Cancer Wisconsin
    dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
-   [SHAP documentation](https://shap.readthedocs.io/)
-   [SHAP DeepExplainer
    documentation](https://shap.readthedocs.io/en/latest/generated/shap.DeepExplainer.html)
-   [Captum DeepLift
    documentation](https://captum.ai/api/deep_lift.html)
-   [DeepLIFT paper -- Shrikumar et
    al.](https://arxiv.org/abs/1704.02685)
-   [SHAP paper -- Lundberg and Lee](https://arxiv.org/abs/1705.07874)
