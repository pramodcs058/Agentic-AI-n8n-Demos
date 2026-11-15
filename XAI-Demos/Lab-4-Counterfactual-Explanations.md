# Lab 4 – Counterfactual Explanations with DiCE-ML

**Platform:** Google Colab  
**Level:** Undergraduate students / beginner to intermediate  
**Estimated duration:** 60–75 minutes  
**Dataset:** German Credit (Credit-G), a public credit-risk dataset hosted on OpenML  
**Libraries:** pandas, scikit-learn, DiCE-ML, Matplotlib

---

## 1. Lab overview

### Learning objectives

By the end of this lab, students will be able to:

1. Explain what a counterfactual explanation is.
2. Distinguish a counterfactual explanation from a feature-importance explanation.
3. Train a simple credit-risk classification model.
4. Use DiCE-ML to generate up to three alternative applicant profiles that change the model's prediction.
5. Compare an original applicant with the generated counterfactuals.
6. Identify limitations: feasibility, actionability, plausibility, fairness, and the difference between a model prediction and a real lending decision.

### Case study: “What could change the prediction?”

Imagine a bank has a machine-learning model that classifies an applicant's credit risk as **good** or **bad**. An applicant receives a `bad` prediction and asks:

> “What would need to be different in my application for this model to predict good credit risk?”

A conventional feature-importance chart might say which features influenced the prediction. A counterfactual explanation instead presents alternative feature values that would change the model's output.

For example, a generated counterfactual might show a different loan duration, credit amount, or savings category. These are **model-generated examples**, not financial advice and not proof that changing a particular attribute will cause a bank to approve a loan.

### What students will build

- A Logistic Regression classifier trained on the German Credit dataset.
- A DiCE-ML explainer connected to that trained model.
- A query applicant predicted as `bad`.
- Up to three counterfactual applicant profiles predicted as `good`.
- A comparison table showing which attributes changed.

### Important teaching distinction

A counterfactual answers:

**“What is a nearby alternative input for which this model gives a different prediction?”**

It does not necessarily answer:

**“What should this person do in real life?”**

The latter requires domain knowledge, legal and ethical review, and an understanding of which attributes are actionable and appropriate to change.

---

## 2. Open Google Colab

1. Open [https://colab.research.google.com/](https://colab.research.google.com/).
2. Select **New notebook**.
3. Rename it `Lab-4-Counterfactual-Explanations`.
4. Select **Runtime → Change runtime type**.
5. Choose **Python 3** and **CPU**. A GPU is not required.
6. Run the cells in order.

---

## 3. Install and import the libraries

### Cell 1 – Install dependencies

DiCE-ML is distributed as the `dice-ml` Python package. The pinned version below is used to keep the demonstration consistent.

```python
%pip install -q dice-ml==0.11 scikit-learn pandas numpy matplotlib
```

If Colab asks you to restart the runtime after installation, choose **Restart session**, then run the import cell again.

### Cell 2 – Import packages

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import dice_ml
from dice_ml import Dice

RANDOM_STATE = 42

print("Imports completed successfully.")
print("DiCE-ML version:", dice_ml.__version__)
```

**Instructor note:** Explain that scikit-learn trains the prediction model, while DiCE-ML searches for alternative inputs that produce a desired prediction from that model.

---

## 4. Load the public German Credit dataset

The German Credit dataset (`credit-g`) contains 1,000 applicant records. The target column is `class`, with values `good` and `bad`. The input features include both numerical and categorical information.

### Cell 3 – Download the dataset from OpenML

```python
credit = fetch_openml(
    name="credit-g",
    version=1,
    as_frame=True
)

X = credit.data.copy()
y = credit.target.astype(str).str.lower()

df = X.copy()
df["class"] = y

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["class"].value_counts())

display(df.head())
```

The first execution downloads the dataset from OpenML, so Colab needs an internet connection.

### Cell 4 – Inspect columns and data types

```python
print("Columns:")
print(df.columns.tolist())

print("\nData types:")
display(df.dtypes.to_frame("data_type"))

print("\nMissing values:")
display(df.isna().sum().to_frame("missing_values"))
```

### Cell 5 – Identify numerical and categorical features

For this dataset, the following columns are numerical. The remaining predictor columns are treated as categorical.

```python
numeric_features = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents"
]

target_column = "class"
feature_columns = [col for col in df.columns if col != target_column]

# Convert numeric columns explicitly to numeric values.
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_features = [
    col for col in feature_columns if col not in numeric_features
]

# Ensure categorical values are consistently represented as strings.
for col in categorical_features:
    df[col] = df[col].astype("string")

# Keep the target as the two string labels: good and bad.
df[target_column] = df[target_column].astype(str).str.lower()

print("Numerical features:", numeric_features)
print("\nCategorical features:", categorical_features)
print("\nTarget labels:", sorted(df[target_column].unique()))
```

### Cell 6 – Check the dataset after preparation

```python
print("Rows and columns:", df.shape)
print("\nMissing values in predictor columns:")
print(df[feature_columns].isna().sum().sort_values(ascending=False))

print("\nTarget counts:")
print(df[target_column].value_counts())
```

**Discuss with students:**

- Why does this dataset contain both numerical and categorical features?
- Why should a category such as `checking_status` not be treated as an ordinary number?
- Why do we need to prepare the data before training a model?

---

## 5. Split the data into training and test sets

The model must be evaluated on records it did not use during training.

### Cell 7 – Train/test split

```python
X = df[feature_columns].copy()
y = df[target_column].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
```

`stratify=y` helps preserve the original proportion of `good` and `bad` outcomes in both subsets.

---

## 6. Build and train the classification model

We will use Logistic Regression. It is a useful teaching model because it is relatively simple, fast to train, and exposes class probabilities. The preprocessing and classifier are placed in a single scikit-learn Pipeline so the same transformations are applied consistently during training and prediction.

### Cell 8 – Define preprocessing

```python
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)
```

### Cell 9 – Create and train the model

```python
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]
)

model.fit(X_train, y_train)

print("Model training completed.")
print("Model classes:", model.classes_)
```

### Cell 10 – Evaluate the model

```python
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)
display.plot(cmap="Blues", values_format="d")
plt.title("German Credit – Test Set Confusion Matrix")
plt.show()
```

**Instructor explanation:**

- `good` and `bad` are the model's class labels, not a complete assessment of a person's creditworthiness.
- Accuracy alone does not tell us the impact of false approvals or false rejections.
- Counterfactual explanations describe this trained model's behaviour. They do not validate the model or make it fair.

---

## 7. Select an applicant for the counterfactual demonstration

We need an applicant for whom the model predicts `bad`. To make the demonstration more likely to find alternatives, we select a `bad` prediction that is relatively close to the model's decision boundary.

### Cell 11 – Find a query applicant

```python
test_probabilities = model.predict_proba(X_test)
good_class_index = list(model.classes_).index("good")

test_results = X_test.copy()
test_results["actual_class"] = y_test
test_results["predicted_class"] = model.predict(X_test)
test_results["probability_good"] = test_probabilities[:, good_class_index]

bad_candidates = test_results[
    test_results["predicted_class"] == "bad"
].copy()

if bad_candidates.empty:
    raise RuntimeError(
        "The model did not predict any test applicant as 'bad'. "
        "Rerun the split/model cell or inspect the model results."
    )

# Select a predicted-bad applicant with the highest probability of good.
# This is a near-boundary example and is often easier to explain.
query_index = bad_candidates["probability_good"].idxmax()

query_instance = X_test.loc[[query_index]].copy()

print("Selected test row:", query_index)
print("Actual class:", y_test.loc[query_index])
print("Predicted class:", model.predict(query_instance)[0])
print(
    "Predicted probability of good:",
    round(model.predict_proba(query_instance)[0][good_class_index], 3)
)

display(query_instance.T.rename(columns={query_index: "Selected applicant"}))
```

**Explain the choice:** We are not selecting an applicant because the actual class is necessarily `bad`. We are selecting an applicant for whom the trained model currently predicts `bad`. The counterfactual task is to find alternative inputs for which the same model predicts `good`.

### Cell 12 – Display the original prediction

```python
original_prediction = model.predict(query_instance)[0]
original_probabilities = model.predict_proba(query_instance)[0]

probability_table = pd.DataFrame({
    "class": model.classes_,
    "predicted_probability": original_probabilities
})

print("Original prediction:", original_prediction)
display(probability_table)
```

---

## 8. Create the DiCE data and model interfaces

DiCE needs two interfaces:

1. **Data interface:** tells DiCE which columns are continuous and which are categorical, and identifies the outcome column.
2. **Model interface:** connects DiCE to the trained scikit-learn model.

We pass the full prepared dataset, including the outcome column, to the data interface. The model interface receives the fitted scikit-learn Pipeline.

### Cell 13 – Configure the DiCE data interface

```python
dice_data = dice_ml.Data(
    dataframe=df[feature_columns + [target_column]].copy(),
    continuous_features=numeric_features,
    outcome_name=target_column
)

print("DiCE data interface created.")
```

### Cell 14 – Configure the DiCE model interface

```python
dice_model = dice_ml.Model(
    model=model,
    backend="sklearn"
)

print("DiCE model interface created.")
```

### Cell 15 – Create the counterfactual explainer

We use DiCE's `random` method for this introductory demonstration. It searches candidate records and checks whether they produce the requested outcome.

```python
explainer = Dice(
    dice_data,
    dice_model,
    method="random"
)

print("DiCE counterfactual explainer is ready.")
```

---

## 9. Generate three counterfactual explanations

We ask DiCE to find up to three alternative profiles for the selected applicant. The desired class is `opposite`, which means the opposite of the model's prediction for the query instance.

For this demonstration, we initially allow DiCE to vary a selected set of potentially changeable application fields. We deliberately leave fields such as age and personal status out of this list. This is a teaching constraint, not a claim that every included field can or should be changed in real life.

### Cell 16 – Choose the features DiCE may vary

```python
features_to_vary = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "checking_status",
    "savings_status",
    "employment",
    "other_payment_plans",
    "housing",
    "existing_credits"
]

print("Features allowed to vary:")
print(features_to_vary)
```

### Cell 17 – Generate up to three counterfactuals

```python
counterfactuals = explainer.generate_counterfactuals(
    query_instances=query_instance,
    total_CFs=3,
    desired_class="opposite",
    features_to_vary=features_to_vary
)

print("Counterfactual generation completed.")
```

DiCE may return fewer than three counterfactuals if it cannot find enough valid candidates under the current constraints. Results can vary because the search is stochastic and depends on the data and fitted model.

### Cell 18 – Display the counterfactual results

```python
counterfactuals.visualize_as_dataframe(
    show_only_changes=True
)
```

**How to read the output:**

- The first profile is the original query applicant.
- The other rows are generated alternatives.
- Only changed attributes are shown when `show_only_changes=True`.
- Each generated alternative is intended to receive the desired model prediction.

---

## 10. Extract and verify the generated counterfactuals

Do not rely only on the visual display. We should independently pass the generated profiles back through the original model and verify their predictions.

### Cell 19 – Extract the generated profiles

```python
cf_example = counterfactuals.cf_examples[0]
final_cfs = cf_example.final_cfs_df

if final_cfs is None or final_cfs.empty:
    raise RuntimeError(
        "DiCE did not find a counterfactual under the current constraints. "
        "See the troubleshooting section and try the suggested alternatives."
    )

# Keep only the original model input columns.
counterfactual_profiles = final_cfs[feature_columns].copy()

print("Number of counterfactuals found:", len(counterfactual_profiles))
display(counterfactual_profiles)
```

### Cell 20 – Verify the model predictions

```python
cf_predictions = model.predict(counterfactual_profiles)
cf_probabilities = model.predict_proba(counterfactual_profiles)

verification = counterfactual_profiles.copy()
verification["model_prediction"] = cf_predictions
verification["probability_good"] = cf_probabilities[:, good_class_index]

display(verification)
```

All returned alternatives should have the requested opposite prediction. If a result does not, inspect the model interface, the output, and the installed DiCE version before presenting it.

### Cell 21 – Compare the original applicant and counterfactuals

```python
comparison = pd.concat(
    [
        query_instance.assign(profile="Original applicant").set_index("profile"),
        counterfactual_profiles.assign(
            profile=[f"Counterfactual {i+1}" for i in range(len(counterfactual_profiles))]
        ).set_index("profile")
    ],
    axis=0
)

display(comparison.T)
```

### Cell 22 – Show only the features that changed

```python
original_row = query_instance.iloc[0]

changed_features = []

for cf_number, (_, cf_row) in enumerate(counterfactual_profiles.iterrows(), start=1):
    for feature in feature_columns:
        original_value = original_row[feature]
        cf_value = cf_row[feature]

        # Compare safely across numeric and categorical values.
        if pd.isna(original_value) and pd.isna(cf_value):
            changed = False
        else:
            changed = str(original_value) != str(cf_value)

        if changed:
            changed_features.append({
                "counterfactual": f"Counterfactual {cf_number}",
                "feature": feature,
                "original_value": original_value,
                "counterfactual_value": cf_value
            })

changed_features_df = pd.DataFrame(changed_features)

if changed_features_df.empty:
    print("No feature differences were found in the returned profiles.")
else:
    display(changed_features_df)
```

**Instructor prompt:** Ask students to identify the smallest set of changes they can see. Then ask whether those changes are realistic, actionable, and fair.

---

## 11. Visualise the model's probability for the original and alternative profiles

A counterfactual is about changing the model's prediction. This chart makes the change in the model's estimated probability visible.

### Cell 23 – Plot probability of the `good` class

```python
profiles_for_plot = pd.concat(
    [query_instance, counterfactual_profiles],
    ignore_index=True
)

profile_names = (
    ["Original applicant"]
    + [f"Counterfactual {i+1}" for i in range(len(counterfactual_profiles))]
)

good_probabilities = model.predict_proba(profiles_for_plot)[:, good_class_index]

plt.figure(figsize=(9, 5))
bars = plt.bar(profile_names, good_probabilities)
plt.axhline(0.5, linestyle="--", label="0.5 reference line")
plt.ylim(0, 1)
plt.ylabel("Model probability of good credit class")
plt.title("Original Applicant vs. Counterfactual Profiles")
plt.xticks(rotation=15, ha="right")
plt.legend()

for bar, value in zip(bars, good_probabilities):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.show()
```

**Teaching note:** The 0.5 line is a visual reference for this binary Logistic Regression classifier. The model's actual class decision is determined by its classification rule and class ordering. Do not generalise this threshold to every model or application.

---

## 12. Experiment: change the counterfactual constraints

Counterfactual generation depends on the features DiCE is allowed to vary. This is a good opportunity to show students that explanations are not unique and that constraints affect the result.

### Experiment A – Allow all features to vary

This is useful for understanding the search, but it may vary attributes that are immutable or inappropriate to change in a real lending context.

```python
all_features_counterfactuals = explainer.generate_counterfactuals(
    query_instances=query_instance,
    total_CFs=3,
    desired_class="opposite",
    features_to_vary="all"
)

all_features_counterfactuals.visualize_as_dataframe(
    show_only_changes=True
)
```

Ask students:

- Did DiCE change any attribute that a person cannot reasonably change?
- Would you accept such a counterfactual as a useful recommendation?
- Why should an explanation system include domain constraints?

### Experiment B – Restrict the features further

Try a smaller set of fields. A smaller search space may make the explanation easier to discuss, but it can also make it impossible to find a counterfactual.

```python
restricted_features = [
    "duration",
    "credit_amount",
    "installment_commitment"
]

restricted_counterfactuals = explainer.generate_counterfactuals(
    query_instances=query_instance,
    total_CFs=3,
    desired_class="opposite",
    features_to_vary=restricted_features
)

restricted_counterfactuals.visualize_as_dataframe(
    show_only_changes=True
)
```

If DiCE cannot find a counterfactual, explain that this means it did not find one under the current search settings and constraints. It does **not** prove that no possible counterfactual exists.

---

## 13. Optional: create a simple prediction function for a new applicant

This optional cell lets students change one or more input values and inspect the model's prediction. Use values and categories observed in the dataset so the example remains within familiar data ranges.

### Cell 24 – Inspect possible values for selected fields

```python
for col in ["checking_status", "savings_status", "employment", "housing"]:
    print(f"\n{col}:")
    print(sorted(df[col].dropna().astype(str).unique().tolist()))
```

### Cell 25 – Make a prediction for a modified profile

Start with the selected applicant and change one value to a valid category or numerical value.

```python
student_applicant = query_instance.copy()

# Example change: reduce the requested duration.
# Replace this with a value suitable for the selected applicant.
student_applicant.loc[:, "duration"] = max(
    4,
    int(query_instance.iloc[0]["duration"]) - 6
)

print("Modified applicant:")
display(student_applicant.T)

print("Predicted class:", model.predict(student_applicant)[0])

student_probabilities = model.predict_proba(student_applicant)[0]
display(pd.DataFrame({
    "class": model.classes_,
    "probability": student_probabilities
}))
```

This is a manual what-if test, not a DiCE counterfactual search. It helps students understand the difference between manually changing an input and asking an algorithm to search for alternative inputs.

---

## 14. Troubleshooting

### Issue 1 – OpenML download fails

**Possible cause:** Colab has no internet connection or OpenML is temporarily unavailable.

**Try:**
- Confirm that the Colab runtime has internet access.
- Rerun the dataset-loading cell.
- If the service is unavailable, retry later. Do not silently replace the dataset without telling students.

### Issue 2 – `ModuleNotFoundError: No module named 'dice_ml'`

Run:

```python
%pip install -q dice-ml==0.11
```

If necessary, restart the Colab runtime and rerun the import cell.

### Issue 3 – DiCE finds fewer than three counterfactuals

This can happen because the search is stochastic and the allowed feature set may be too restrictive.

Try the following in order:

1. Keep the original query applicant and rerun the generation cell.
2. Increase the allowed features by adding another plausible field from the dataset.
3. Try `features_to_vary="all"` as a diagnostic experiment.
4. Reduce `total_CFs` to `1` to check whether any counterfactual can be found.
5. Explain that a failed search is not proof that a counterfactual does not exist.

For example:

```python
one_counterfactual = explainer.generate_counterfactuals(
    query_instances=query_instance,
    total_CFs=1,
    desired_class="opposite",
    features_to_vary="all"
)

one_counterfactual.visualize_as_dataframe(
    show_only_changes=True
)
```

### Issue 4 – A generated counterfactual looks unrealistic

DiCE searches within the supplied data and the constraints provided, but that does not guarantee every combination is realistic or actionable. Discuss the need for:
- Immutable-feature constraints.
- Valid ranges and domain rules.
- Plausible combinations of categorical and numerical values.
- Human review before using an explanation to guide a real decision.

### Issue 5 – A warning appears during model training

Warnings about convergence or feature names should be read rather than ignored in production work. This lab uses `max_iter=2000`; if convergence warnings remain, inspect the data and model configuration. Do not treat a warning-free run as proof that the model is suitable for lending decisions.

---

## 15. Key concepts to reinforce

| Concept | Explanation |
|---|---|
| Factual instance | The original applicant record being explained |
| Counterfactual instance | An alternative input that changes the model's prediction |
| Desired class | The prediction DiCE is asked to reach |
| Features to vary | The attributes DiCE is permitted to change |
| Immutable feature | An attribute that should not be changed in the counterfactual search |
| Actionability | Whether a person can realistically and appropriately change an attribute |
| Plausibility | Whether the generated profile is realistic and consistent with the domain |
| Proximity | How different the counterfactual is from the original instance |
| Diversity | How different the generated counterfactuals are from one another |

### Counterfactual vs. feature importance

| Feature importance | Counterfactual explanation |
|---|---|
| Describes which features influenced a prediction | Shows alternative input values that change a prediction |
| Often gives an importance score | Gives one or more alternative profiles |
| Answers “Which features mattered?” | Answers “What could be different?” |
| Does not necessarily suggest an alternative input | Explicitly searches for an alternative input |

---

## 16. Limitations and responsible-use discussion

This dataset is a teaching dataset. It is not a current bank's lending policy, and the model is not a validated credit decision system.

Emphasise these points:

1. **A counterfactual is model-specific.** A different model may produce different alternatives.
2. **A counterfactual is not causal evidence.** If changing a field changes a prediction, that does not prove that making the same change in real life will cause the same outcome.
3. **Actionability matters.** Some attributes cannot be changed; others may be inappropriate to use as recommendations.
4. **Plausibility matters.** A mathematically valid combination can still be unrealistic.
5. **Fairness matters.** Sensitive or proxy attributes can create harmful explanations. Do not use counterfactual output as an instruction to manipulate protected or personal characteristics.
6. **Model quality matters.** An explanation of a poor model does not make the model trustworthy.
7. **Human review matters.** High-impact financial decisions require appropriate governance, validation, and human oversight.

---

## 17. Student activities

### Activity 1 – Interpret the explanation

For the selected applicant, write down:
- The original predicted class.
- The number of counterfactuals returned.
- The features that changed in each counterfactual.
- Whether each change seems actionable and plausible.

### Activity 2 – Change the search constraints

Run the explanation once with the restricted feature list and once with all features.

Answer:
- Did the number of counterfactuals change?
- Did the changed attributes differ?
- Which result is easier to explain to an applicant?
- Which result is more realistic?

### Activity 3 – Test another applicant

Select a different applicant predicted as `bad`, then generate counterfactuals for that applicant.

Hint:

```python
another_query = bad_candidates.sample(
    n=1,
    random_state=7
)[feature_columns]

another_query
```

Then pass `another_query` to `explainer.generate_counterfactuals()` using the same settings.

### Activity 4 – Think critically

A counterfactual says that a different `credit_amount` would change the model prediction.

Discuss:
- Does that mean the applicant should request a smaller loan?
- Could the model be learning a correlation rather than a causal relationship?
- What additional information would a bank need before making a recommendation?

---

## 18. Knowledge-check questions

1. What is a counterfactual explanation?
2. What is the difference between the original applicant and a counterfactual applicant?
3. What does `desired_class="opposite"` ask DiCE to do?
4. Why do we specify `features_to_vary`?
5. Why should immutable attributes be excluded from a real-world counterfactual search?
6. Does a counterfactual prove that changing an attribute will cause a real-world outcome to change?
7. Why might DiCE return fewer counterfactuals than requested?
8. How is a counterfactual explanation different from SHAP feature importance?

### Suggested answers

1. It is an alternative input for which the model produces a different or specified prediction.
2. The original is the record being explained; the counterfactual is a generated alternative.
3. It asks for the opposite of the model's prediction on the query instance.
4. It constrains the search to selected input features.
5. Because some attributes cannot or should not be changed, and changing them can produce inappropriate explanations.
6. No. It describes the behaviour of the model, not a causal effect in the real world.
7. The search may not find enough candidates under the chosen constraints and settings.
8. SHAP estimates feature contributions to a prediction; a counterfactual searches for alternative feature values that change the prediction.

---

## 19. Suggested 60–75 minute instructor flow

| Time | Demonstration |
|---|---|
| 0–5 min | Introduce the bank applicant case study and counterfactual question |
| 5–15 min | Load and inspect the German Credit dataset |
| 15–25 min | Prepare data, train the Logistic Regression model, and evaluate it |
| 25–35 min | Select an applicant and inspect the original prediction |
| 35–50 min | Configure DiCE and generate up to three counterfactuals |
| 50–60 min | Compare changed features and verify the new predictions |
| 60–70 min | Run the feature-constraint experiment and discuss actionability |
| 70–75 min | Knowledge check and recap |

For a 60-minute session, treat the optional new-applicant prediction and some student activities as take-home exercises.

---

## 20. Closing explanation for students

Use this summary to conclude the demonstration:

> “A prediction tells us what the model decided. A feature-importance explanation helps us understand which inputs influenced that decision. A counterfactual explanation takes the next step: it searches for an alternative version of the input that would change the model's prediction. But remember, a model-generated alternative is not automatically realistic, fair, actionable, or causal. Explainability is useful only when we interpret it responsibly.”

### Transition to the next lab

In this lab, we asked: **“What would need to be different for the model to predict another class?”**

In the next lab on **Integrated Gradients for LSTMs**, we will move from tabular applicant records to time-series data and ask:

**“Which time steps in a sequence contributed most to the model's prediction?”**

---

## References

- OpenML – German Credit (`credit-g`): [https://www.openml.org/search?type=data&sort=runs&id=31](https://www.openml.org/search?type=data&sort=runs&id=31)
- DiCE-ML GitHub repository: [https://github.com/interpretml/DiCE](https://github.com/interpretml/DiCE)
- DiCE documentation: [https://interpret.ml/DiCE/](https://interpret.ml/DiCE/)
- scikit-learn documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

**End of Lab 4**
