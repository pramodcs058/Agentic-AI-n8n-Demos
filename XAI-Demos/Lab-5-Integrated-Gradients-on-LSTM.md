# Lab 5 – Integrated Gradients for LSTM Time-Series Forecasting

**Platform:** Google Colab  
**Level:** Undergraduate students / beginner to intermediate  
**Estimated duration:** 60–75 minutes  
**Dataset:** UCI Individual Household Electric Power Consumption  
**Model:** PyTorch LSTM regression model  
**Explainability library:** Captum Integrated Gradients  
**Task:** Predict the next hour's household electricity consumption and attribute the prediction to the previous 24 hourly observations.

---

## 1. Lab overview

### Learning objectives

By the end of this lab, students will be able to:

1. Explain why time-series models need explanations that consider time order.
2. Prepare a real electricity-consumption time series for supervised learning.
3. Train a basic Long Short-Term Memory (LSTM) network to forecast the next hour's electricity consumption.
4. Explain the Integrated Gradients (IG) method and the role of a baseline.
5. Use Captum to calculate attribution scores for the 24 time steps supplied to an LSTM.
6. Read a temporal attribution plot and identify time steps that increase or decrease the model's forecast relative to the selected baseline.
7. Explain the limitations of attribution scores, including baseline sensitivity and the difference between model attribution and causality.

### Case study: “Which hours influenced the forecast?”

Imagine a smart-energy application that predicts a household's electricity consumption for the next hour. It receives the previous 24 hours of readings and passes them to an LSTM.

The application predicts the next hour's electricity use. A user or energy analyst asks:

> “Which parts of the last 24 hours influenced this forecast the most?”

Integrated Gradients helps us assign an attribution score to each input time step. A positive attribution indicates that the time step pushes the model output upward relative to the chosen baseline. A negative attribution indicates that it pushes the output downward relative to the baseline.

### Dataset selection

This lab uses the **UCI Individual Household Electric Power Consumption** dataset. It contains approximately 2.1 million one-minute observations collected in one household in Sceaux, France, from December 2006 to November 2010. The dataset includes `Global_active_power`, measured in kilowatts, as well as other electrical measurements.

For a training demonstration, we will:
- Use only `Global_active_power` to keep the example focused.
- Resample the minute-level readings into hourly averages.
- Use the previous 24 hourly values to predict the next hourly value.
- Split the series chronologically into training, validation, and test periods.
- Fit the scaling statistics using the training period only.

The dataset contains missing readings. The notebook handles missing hourly values and avoids creating a training window when the input sequence or target contains a missing value.

### What students will build

- An hourly electricity-consumption time series.
- A small LSTM model that forecasts one hour ahead.
- A test prediction with its preceding 24-hour input window.
- Integrated Gradients attributions for the 24 input time steps.
- A bar chart showing positive and negative temporal attributions.
- A completeness check comparing the sum of attributions with the difference between the model outputs for the input and baseline.

---

## 2. Open Google Colab

1. Open [https://colab.research.google.com/](https://colab.research.google.com/).
2. Select **New notebook**.
3. Rename it `Lab-5-Integrated-Gradients-on-LSTM`.
4. Select **Runtime → Change runtime type**.
5. Choose **Python 3**. A CPU runtime is sufficient for this small LSTM.
6. Run the cells in order.

The dataset download requires an internet connection. The first download may take a short while because the original archive is approximately 20 MB.

---

## 3. Install and import the libraries

### Cell 1 – Install Captum

Colab generally includes PyTorch. We install Captum and the data-handling libraries used in this lab.

```python
%pip install -q captum==0.7.0 pandas numpy matplotlib scikit-learn
```

If Colab asks for a runtime restart, restart the session and then run the import cell.

### Cell 2 – Import packages

```python
import io
import os
import random
import zipfile
import requests
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from captum.attr import IntegratedGradients

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("Device:", device)
print("Imports completed.")
```

**Instructor note:** Captum is a model-interpretability library for PyTorch. It provides Integrated Gradients as an attribution method that can be applied to the input tensor of our LSTM.

---

## 4. Download and load the UCI electricity dataset

The UCI archive contains a semicolon-separated text file. The missing readings are represented by `?`.

### Cell 3 – Download the ZIP archive

```python
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

response = requests.get(DATA_URL, timeout=120)
response.raise_for_status()

print("Downloaded archive size:", round(len(response.content) / (1024 * 1024), 2), "MB")
```

### Cell 4 – Read the text file directly from the ZIP archive

```python
with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
    text_files = [
        name for name in archive.namelist()
        if name.lower().endswith("household_power_consumption.txt")
    ]

    if not text_files:
        raise FileNotFoundError(
            "Could not find household_power_consumption.txt in the UCI archive."
        )

    with archive.open(text_files[0]) as file:
        raw_df = pd.read_csv(
            file,
            sep=";",
            na_values=["?"],
            low_memory=False
        )

raw_df.columns = raw_df.columns.str.strip()

print("Raw dataset shape:", raw_df.shape)
display(raw_df.head())
```

### Cell 5 – Inspect the dataset

```python
print("Columns:")
print(raw_df.columns.tolist())

print("\nMissing values in selected measurement:")
print(raw_df["Global_active_power"].isna().sum())

print("\nData types:")
display(raw_df.dtypes.to_frame("data_type"))
```

The column `Global_active_power` represents the household's minute-averaged active power in kilowatts.

---

## 5. Prepare the hourly time series

We combine the `Date` and `Time` columns into one timestamp, convert the power readings to numeric values, and resample them into hourly averages.

### Cell 6 – Create the timestamp

```python
raw_df["timestamp"] = pd.to_datetime(
    raw_df["Date"].astype(str).str.strip()
    + " "
    + raw_df["Time"].astype(str).str.strip(),
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

raw_df["Global_active_power"] = pd.to_numeric(
    raw_df["Global_active_power"],
    errors="coerce"
)

raw_df = raw_df.dropna(subset=["timestamp"])
raw_df = raw_df.set_index("timestamp").sort_index()

print("First timestamp:", raw_df.index.min())
print("Last timestamp:", raw_df.index.max())
```

### Cell 7 – Resample minute readings into hourly averages

```python
hourly_power = (
    raw_df["Global_active_power"]
    .resample("h")
    .mean()
    .rename("global_active_power_kw")
)

print("Hourly observations:", len(hourly_power))
print("Missing hourly values before interpolation:", hourly_power.isna().sum())

display(hourly_power.head())
```

### Cell 8 – Handle missing hourly readings

We interpolate short gaps of up to six consecutive hourly values. Longer gaps are left missing. Later, the window-generation function will skip any sequence containing a missing value, so the LSTM is not trained on incomplete windows.

```python
hourly_power = hourly_power.interpolate(
    method="time",
    limit=6,
    limit_direction="both"
)

print("Missing hourly values after limited interpolation:", hourly_power.isna().sum())

plt.figure(figsize=(14, 4))
hourly_power.iloc[:24 * 7].plot()
plt.title("Hourly Household Electricity Consumption – First Week")
plt.xlabel("Time")
plt.ylabel("Global active power (kW)")
plt.tight_layout()
plt.show()
```

**Discuss with students:**

- Why do we convert minute readings into hourly averages?
- Why is the timestamp important in time-series data?
- Why should we not randomly shuffle the original time series before splitting it?

---

## 6. Split the series chronologically and standardise it

For time-series forecasting, training data should occur before validation and test data. A random split could allow future observations to influence the training process.

We use:
- First 70%: training period
- Next 15%: validation period
- Final 15%: test period

The mean and standard deviation are calculated from the training period only. This avoids using information from the validation or test periods when scaling the data.

### Cell 9 – Chronological split

```python
n_total = len(hourly_power)

train_end = int(n_total * 0.70)
val_end = int(n_total * 0.85)

train_raw = hourly_power.iloc[:train_end].copy()
val_raw = hourly_power.iloc[train_end:val_end].copy()
test_raw = hourly_power.iloc[val_end:].copy()

print("Training period:", train_raw.index.min(), "to", train_raw.index.max())
print("Validation period:", val_raw.index.min(), "to", val_raw.index.max())
print("Test period:", test_raw.index.min(), "to", test_raw.index.max())

print("\nObservations:")
print("Train:", len(train_raw))
print("Validation:", len(val_raw))
print("Test:", len(test_raw))
```

### Cell 10 – Standardise using training statistics

```python
train_mean = float(train_raw.mean())
train_std = float(train_raw.std())

if not np.isfinite(train_std) or train_std == 0:
    raise ValueError("Training standard deviation is invalid.")

train_scaled = (train_raw - train_mean) / train_std
val_scaled = (val_raw - train_mean) / train_std
test_scaled = (test_raw - train_mean) / train_std

print("Training mean (kW):", round(train_mean, 4))
print("Training standard deviation (kW):", round(train_std, 4))
```

**Instructor explanation:** Standardisation makes the numerical scale easier for the neural network to learn. A scaled value of zero represents the average electricity consumption in the training period. This becomes useful when we define the Integrated Gradients baseline.

---

## 7. Convert the time series into LSTM input windows

An LSTM expects sequential input. We will use the previous 24 hourly measurements to predict the next hour.

Example:

| Input sequence | Prediction target |
|---|---|
| Hour 1 to Hour 24 | Hour 25 |
| Hour 2 to Hour 25 | Hour 26 |
| Hour 3 to Hour 26 | Hour 27 |

Our input tensor will have this shape:

`[number_of_samples, 24, 1]`

- `number_of_samples`: number of available windows.
- `24`: number of hourly time steps.
- `1`: one input feature, active power.

### Cell 11 – Define a window-generation function

```python
LOOKBACK = 24

def create_windows(series, lookback=LOOKBACK):
    """
    Create input windows of length `lookback` and the next-step target.

    Windows containing missing or non-finite values are skipped.
    The function preserves the timestamps of both the input window
    and the prediction target.
    """
    values = series.to_numpy(dtype=np.float32)
    timestamps = series.index

    X_windows = []
    y_targets = []
    target_times = []
    input_times = []

    for start in range(len(values) - lookback):
        end = start + lookback

        input_window = values[start:end]
        target_value = values[end]

        if not np.isfinite(input_window).all():
            continue

        if not np.isfinite(target_value):
            continue

        X_windows.append(input_window.reshape(lookback, 1))
        y_targets.append(target_value)
        target_times.append(timestamps[end])
        input_times.append(timestamps[start:end])

    X_windows = np.asarray(X_windows, dtype=np.float32)
    y_targets = np.asarray(y_targets, dtype=np.float32)

    return X_windows, y_targets, pd.DatetimeIndex(target_times), input_times
```

### Cell 12 – Create train, validation, and test windows

We construct the windows within each chronological partition. No window crosses from the training period into validation or from validation into test.

```python
X_train, y_train, train_target_times, train_input_times = create_windows(train_scaled)
X_val, y_val, val_target_times, val_input_times = create_windows(val_scaled)
X_test, y_test, test_target_times, test_input_times = create_windows(test_scaled)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
```

Expected input shape is similar to:

`(number_of_windows, 24, 1)`

The exact number of windows depends on the missing readings that were skipped.

### Cell 13 – Convert arrays to PyTorch tensors

```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("Tensor shape:", X_train_tensor.shape)
```

---

## 8. Build the LSTM forecasting model

An LSTM is a recurrent neural network designed to learn patterns from sequential data. It maintains an internal state as it processes the time steps.

For this lab:
- Input size: 1 feature (`Global_active_power`)
- Hidden size: 32
- Number of LSTM layers: 1
- Output: one numerical value representing the next-hour forecast in standardised units

### Cell 14 – Define the LSTM model

```python
class ElectricityLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        # Use the output corresponding to the final time step.
        final_time_step = lstm_output[:, -1, :]

        prediction = self.regressor(final_time_step)

        # Return shape [batch_size], a scalar output per example.
        return prediction.squeeze(-1)


model = ElectricityLSTM(
    input_size=1,
    hidden_size=32,
    num_layers=1
).to(device)

print(model)
```

### Cell 15 – Create DataLoaders

```python
BATCH_SIZE = 128

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Training batches:", len(train_loader))
```

We shuffle the already-created training windows during model optimisation. This does not change their internal time order; each 24-hour sequence remains chronological. The original train, validation, and test partitions remain chronological and separate.

---

## 9. Train the LSTM

We use Mean Squared Error (MSE) as the regression loss and Adam as the optimiser.

### Cell 16 – Configure training

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 8
```

### Cell 17 – Define validation evaluation

```python
def calculate_loss(model, X_tensor, y_tensor):
    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor.to(device))
        targets = y_tensor.to(device)
        loss = criterion(predictions, targets)

    return float(loss.item())
```

### Cell 18 – Train the model

```python
history = {
    "train_loss": [],
    "val_loss": []
}

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    total_examples = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        loss.backward()

        # Helps prevent unusually large gradients during recurrent training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * len(batch_X)
        total_examples += len(batch_X)

    train_loss = running_loss / total_examples
    val_loss = calculate_loss(model, X_val_tensor, y_val_tensor)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.5f} | "
        f"Validation MSE: {val_loss:.5f}"
    )
```

### Cell 19 – Plot the training and validation losses

```python
plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], label="Training loss")
plt.plot(history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("LSTM Training History")
plt.legend()
plt.tight_layout()
plt.show()
```

**Instructor note:** The training and validation loss curves help students check whether the model is learning. A low training loss with a much higher validation loss can indicate overfitting.

---

## 10. Evaluate the LSTM on the test period

The model predicts standardised values. We convert the predictions back to kilowatts before calculating the final MAE and RMSE.

### Cell 20 – Generate test predictions

```python
model.eval()

with torch.no_grad():
    test_predictions_scaled = (
        model(X_test_tensor.to(device))
        .cpu()
        .numpy()
    )

# Reverse standardisation to get values in kW.
test_predictions_kw = test_predictions_scaled * train_std + train_mean
test_actual_kw = y_test * train_std + train_mean

mae = mean_absolute_error(test_actual_kw, test_predictions_kw)
rmse = np.sqrt(mean_squared_error(test_actual_kw, test_predictions_kw))

print(f"Test MAE:  {mae:.4f} kW")
print(f"Test RMSE: {rmse:.4f} kW")
```

### Cell 21 – Plot actual versus predicted consumption

```python
plot_count = min(24 * 7, len(test_actual_kw))

plt.figure(figsize=(14, 5))
plt.plot(
    test_target_times[:plot_count],
    test_actual_kw[:plot_count],
    label="Actual consumption"
)
plt.plot(
    test_target_times[:plot_count],
    test_predictions_kw[:plot_count],
    label="LSTM prediction"
)
plt.xlabel("Time")
plt.ylabel("Global active power (kW)")
plt.title("Actual vs. Predicted Electricity Consumption – First Test Week")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
```

**Discuss with students:**

- Is the model able to follow the broad consumption pattern?
- Does it miss sudden peaks?
- Why is it useful to understand what parts of the input sequence influence a forecast?

---

## 11. Select one test example for explanation

We select a deterministic example from the test set. The model receives the previous 24 hours and predicts the next hour.

### Cell 22 – Select an example

```python
if len(X_test) == 0:
    raise RuntimeError("No valid test windows were created.")

sample_index = len(X_test) // 2

input_tensor = torch.tensor(
    X_test[sample_index:sample_index + 1],
    dtype=torch.float32,
    device=device
)

selected_input_times = test_input_times[sample_index]
selected_target_time = test_target_times[sample_index]

input_values_kw = (
    X_test[sample_index, :, 0] * train_std + train_mean
)

actual_next_hour_kw = (
    y_test[sample_index] * train_std + train_mean
)

model.eval()

with torch.no_grad():
    predicted_next_hour_scaled = model(input_tensor).item()

predicted_next_hour_kw = (
    predicted_next_hour_scaled * train_std + train_mean
)

print("Forecast target time:", selected_target_time)
print(f"Predicted next-hour consumption: {predicted_next_hour_kw:.3f} kW")
print(f"Actual next-hour consumption:    {actual_next_hour_kw:.3f} kW")
```

### Cell 23 – Plot the 24-hour input and forecast

```python
plt.figure(figsize=(12, 4))
plt.plot(
    selected_input_times,
    input_values_kw,
    marker="o",
    label="Previous 24 hours"
)
plt.scatter(
    [selected_target_time],
    [predicted_next_hour_kw],
    marker="X",
    s=100,
    label="LSTM forecast"
)
plt.scatter(
    [selected_target_time],
    [actual_next_hour_kw],
    marker="o",
    s=70,
    label="Actual next hour"
)

plt.xlabel("Time")
plt.ylabel("Global active power (kW)")
plt.title("The 24-Hour Input Window and Next-Hour Forecast")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.show()
```

**Teaching prompt:** The LSTM receives 24 readings, not just the last reading. Ask students why a sequence of observations might provide more information than a single observation.

---

## 12. Understand Integrated Gradients

Integrated Gradients assigns an attribution to each input element by accumulating gradients along a straight-line path from a baseline input to the actual input.

For an input \(x\), baseline \(x'\), and model output \(F\), the attribution for input element \(i\) is:

\[
IG_i(x) = (x_i - x'_i)
\int_0^1
\frac{\partial F(x' + \alpha(x-x'))}{\partial x_i}
\,d\alpha
\]

In this lab:
- `x` is the 24-hour input sequence.
- `x'` is the baseline sequence.
- `F` is the LSTM's next-hour forecast.
- Each input element corresponds to one hourly observation.

### What baseline will we use?

Our inputs are standardised using the training mean and standard deviation. Therefore, a scaled value of zero represents the **average hourly consumption in the training period**.

We use an all-zero baseline:

```python
baseline_tensor = torch.zeros_like(input_tensor)
```

This means we are asking:

> “Compared with a sequence where every one of the previous 24 hours is at the training-period average consumption, which actual hourly observations move the forecast up or down?”

The baseline is part of the explanation. A different baseline can produce different attribution scores, so students should not treat the scores as absolute or baseline-independent truth.

---

## 13. Calculate Integrated Gradients with Captum

Captum's `IntegratedGradients` takes the model as its forward function. Our LSTM returns one scalar forecast per example, so we do not need to specify a classification target index.

### Cell 24 – Create the baseline

```python
baseline_tensor = torch.zeros_like(input_tensor)

print("Input shape:", tuple(input_tensor.shape))
print("Baseline shape:", tuple(baseline_tensor.shape))
```

Both tensors have shape:

`[1, 24, 1]`

### Cell 25 – Run Integrated Gradients

```python
model.eval()

integrated_gradients = IntegratedGradients(model)

attributions, convergence_delta = integrated_gradients.attribute(
    inputs=input_tensor,
    baselines=baseline_tensor,
    n_steps=100,
    method="gausslegendre",
    internal_batch_size=100,
    return_convergence_delta=True
)

print("Attribution tensor shape:", tuple(attributions.shape))
print("Convergence delta:", float(convergence_delta.detach().cpu().item()))
```

The returned attribution tensor has the same shape as the input: one attribution for each of the 24 time steps.

**Instructor explanation:**

- `n_steps=100` controls how many points are used to approximate the integral.
- `method="gausslegendre"` selects the numerical integration method.
- `return_convergence_delta=True` asks Captum to report a completeness-related approximation error.
- A smaller absolute delta generally indicates that the numerical approximation better satisfies the completeness property. It is not a model-accuracy score.

---

## 14. Inspect the 24 temporal attribution values

### Cell 26 – Convert attributions into a DataFrame

```python
temporal_attributions = (
    attributions
    .detach()
    .squeeze(0)
    .squeeze(-1)
    .cpu()
    .numpy()
)

attribution_df = pd.DataFrame({
    "timestamp": selected_input_times,
    "hourly_consumption_kw": input_values_kw,
    "integrated_gradient": temporal_attributions
})

attribution_df["absolute_attribution"] = (
    attribution_df["integrated_gradient"].abs()
)

display(attribution_df)
```

### Cell 27 – Identify the five most influential time steps by absolute attribution

```python
top_time_steps = attribution_df.sort_values(
    "absolute_attribution",
    ascending=False
).head(5)

display(top_time_steps)
```

A high absolute attribution means that the time step has a relatively large contribution to the difference between the model output for the actual input and the baseline, according to this attribution method and baseline.

---

## 15. Visualise positive and negative attributions

### Cell 28 – Plot the Integrated Gradients time-step scores

```python
plt.figure(figsize=(14, 5))

plt.bar(
    attribution_df["timestamp"].dt.strftime("%m-%d %H:%M"),
    attribution_df["integrated_gradient"]
)

plt.axhline(0, linewidth=1)
plt.xlabel("Input time step")
plt.ylabel("Integrated Gradients attribution")
plt.title("Integrated Gradients – Contribution of Each Hour to the Forecast")
plt.xticks(rotation=65)
plt.tight_layout()
plt.show()
```

### How to explain this plot

- **Positive score:** That hour moves the model's forecast upward relative to the average-consumption baseline.
- **Negative score:** That hour moves the model's forecast downward relative to the average-consumption baseline.
- **Large absolute score:** That hour has a relatively large attribution for this particular forecast and baseline.
- **Score near zero:** That hour has a small attribution under this particular explanation setup.

Do not say that a positive attribution means the hour caused higher real-world consumption. It only describes how the trained LSTM's output changes along the Integrated Gradients path from the baseline to the actual sequence.

---

## 16. Check the completeness property

Integrated Gradients is designed so that the sum of the attributions approximates the difference between the model output at the actual input and the model output at the baseline.

\[
\sum_i IG_i(x) \approx F(x) - F(x')
\]

We can calculate both sides and compare them.

### Cell 29 – Verify the attribution sum

```python
model.eval()

with torch.no_grad():
    output_for_input = model(input_tensor).item()
    output_for_baseline = model(baseline_tensor).item()

attribution_sum = float(attributions.detach().sum().cpu().item())
output_difference = output_for_input - output_for_baseline
absolute_completeness_error = abs(attribution_sum - output_difference)

print(f"Model output for actual input (scaled): {output_for_input:.6f}")
print(f"Model output for baseline (scaled):      {output_for_baseline:.6f}")
print(f"Difference in model outputs:             {output_difference:.6f}")
print(f"Sum of Integrated Gradients:              {attribution_sum:.6f}")
print(f"Absolute completeness error:              {absolute_completeness_error:.8f}")
print(
    "Captum convergence delta:                 ",
    float(convergence_delta.detach().cpu().item())
)
```

The output values and attribution sum are in standardised model-output units. We use the scaled output here because Integrated Gradients is calculated on the LSTM's standardised target.

**Instructor note:** This is a numerical consistency check, not proof that the explanation is correct in every conceptual or causal sense.

---

## 17. Experiment with a different baseline

The baseline affects what the attribution means. Our first baseline represents the training-period average at every hour.

Now we try a second baseline: a **flat sequence equal to the average consumption of this particular 24-hour input window**. This asks a slightly different question:

> “Which hours in this particular day differ from that day's own average pattern in ways that influence the forecast?”

### Cell 30 – Create a window-mean baseline

```python
window_mean_baseline = (
    input_tensor.mean(dim=1, keepdim=True)
    .expand_as(input_tensor)
)

print("Window-mean baseline shape:", tuple(window_mean_baseline.shape))
```

### Cell 31 – Calculate Integrated Gradients with the alternative baseline

```python
window_mean_attributions, window_mean_delta = integrated_gradients.attribute(
    inputs=input_tensor,
    baselines=window_mean_baseline,
    n_steps=100,
    method="gausslegendre",
    internal_batch_size=100,
    return_convergence_delta=True
)

alternative_scores = (
    window_mean_attributions
    .detach()
    .squeeze(0)
    .squeeze(-1)
    .cpu()
    .numpy()
)

print(
    "Alternative-baseline convergence delta:",
    float(window_mean_delta.detach().cpu().item())
)
```

### Cell 32 – Compare the two baselines

```python
baseline_comparison = pd.DataFrame({
    "timestamp": selected_input_times,
    "training_mean_baseline_IG": temporal_attributions,
    "window_mean_baseline_IG": alternative_scores
})

display(baseline_comparison)

plt.figure(figsize=(14, 5))
plt.plot(
    baseline_comparison["timestamp"],
    baseline_comparison["training_mean_baseline_IG"],
    marker="o",
    label="Training-period mean baseline"
)
plt.plot(
    baseline_comparison["timestamp"],
    baseline_comparison["window_mean_baseline_IG"],
    marker="x",
    label="Input-window mean baseline"
)
plt.axhline(0, linewidth=1)
plt.xlabel("Input time step")
plt.ylabel("Integrated Gradients attribution")
plt.title("How the Baseline Changes Temporal Attributions")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
```

**Discuss with students:**

- Did the attribution values change?
- Did the most influential time steps change?
- Which baseline is more appropriate for the question being asked?
- Why should an analyst state the baseline when presenting an attribution chart?

---

## 18. Optional experiment: increase the number of integration steps

A larger `n_steps` can improve the numerical approximation, at the cost of additional computation.

### Cell 33 – Try 200 integration steps

```python
attributions_200, delta_200 = integrated_gradients.attribute(
    inputs=input_tensor,
    baselines=baseline_tensor,
    n_steps=200,
    method="gausslegendre",
    internal_batch_size=200,
    return_convergence_delta=True
)

print("Delta with 100 steps:", float(convergence_delta.detach().cpu().item()))
print("Delta with 200 steps:", float(delta_200.detach().cpu().item()))
```

Ask students whether the convergence delta changed. Explain that a smaller delta is useful evidence about the numerical approximation, but the number of steps should be chosen with both computational cost and explanation stability in mind.

---

## 19. Troubleshooting

### Issue 1 – Dataset download fails

**Possible cause:** Colab cannot reach the UCI server, or the server is temporarily unavailable.

**Try:**
- Check that the runtime has internet access.
- Rerun the download cell.
- Confirm that the UCI dataset page is available.
- Avoid changing the dataset silently because the lab's expected column names and preparation steps depend on this source.

### Issue 2 – `BadZipFile` or text file not found

The archive may not have downloaded correctly. Check the response status and size:

```python
print("HTTP status:", response.status_code)
print("Downloaded bytes:", len(response.content))
```

Rerun the download cell if the response is incomplete.

### Issue 3 – `KeyError: 'Global_active_power'`

Inspect the column names:

```python
print(raw_df.columns.tolist())
```

The code strips whitespace from column names. If the source file changes, identify the exact active-power column before proceeding.

### Issue 4 – No valid windows are available

Inspect the missing hourly readings:

```python
print("Hourly values:", len(hourly_power))
print("Missing values:", hourly_power.isna().sum())
print(hourly_power[hourly_power.isna()].head(20))
```

A window is skipped when any of its 24 input values or its next-hour target is missing. If a large section is missing, inspect the resampling and interpolation results before continuing.

### Issue 5 – Training is slow

For a classroom demonstration:
- Keep `LOOKBACK = 24`.
- Reduce `EPOCHS` from `8` to `4`.
- Reduce `BATCH_SIZE` from `128` to `64` if memory is limited.
- Use a GPU runtime only if available; it is not required for this model.

### Issue 6 – GPU/device mismatch

Ensure the model and input tensors use the same device. In this lab, the model is moved to `device`, and the training and explanation input tensors are also moved to `device`.

### Issue 7 – Integrated Gradients takes too long

Start with `n_steps=50`, then increase to 100 or 200. The number of integration steps controls the numerical approximation work.

### Issue 8 – The attribution scores are difficult to interpret

Check:
- Which baseline was used.
- Whether the input is standardised.
- Whether the plotted values are signed or absolute.
- Whether students are interpreting the score as a model attribution rather than a causal effect.

---

## 20. Key concepts to reinforce

| Concept | Explanation |
|---|---|
| Time series | Observations recorded in time order |
| LSTM | A recurrent neural network that processes sequential inputs |
| Lookback window | The previous observations supplied to the model |
| Forecast horizon | How far ahead the model predicts; this lab predicts one hour ahead |
| Baseline | The reference input from which Integrated Gradients starts |
| Attribution | A score assigned to an input element to explain a model output relative to the baseline |
| Positive attribution | Moves the model output upward relative to the baseline |
| Negative attribution | Moves the model output downward relative to the baseline |
| Integration steps | Numerical points used to approximate the path integral |
| Completeness | The sum of attributions should approximate the difference between output at the input and output at the baseline |
| Convergence delta | Captum's estimate of the approximation error for the completeness property |

### Integrated Gradients vs. feature importance for tabular data

| Feature importance | Integrated Gradients on a time series |
|---|---|
| Often attributes importance to input features | Attributes importance to input elements, here the time steps |
| May not preserve temporal context | Works with the ordered input sequence supplied to the LSTM |
| Answers “Which features mattered?” | Answers “Which time steps contributed to this forecast relative to a baseline?” |

---

## 21. Student activities

### Activity 1 – Interpret the time-step chart

For the selected test example, record:
- The forecasted next-hour electricity consumption.
- The actual next-hour electricity consumption.
- The three time steps with the largest absolute attribution.
- Whether each of those three attributions is positive or negative.

### Activity 2 – Change the baseline

Compare the training-mean baseline and the input-window-mean baseline.

Answer:
- Which time steps changed the most?
- Did any attribution change sign?
- How does the baseline affect the explanation?

### Activity 3 – Explain another test forecast

Change the selected index and run the explanation again.

```python
sample_index = min(100, len(X_test) - 1)

input_tensor = torch.tensor(
    X_test[sample_index:sample_index + 1],
    dtype=torch.float32,
    device=device
)

selected_input_times = test_input_times[sample_index]
selected_target_time = test_target_times[sample_index]

input_values_kw = X_test[sample_index, :, 0] * train_std + train_mean
```

Rerun the cells that create the baseline, calculate Integrated Gradients, and plot the attributions.

### Activity 4 – Think critically

Suppose the largest positive attribution comes from an observation at 7 PM.

Discuss:
- Does this prove that electricity use at 7 PM caused the forecast to be high?
- Could the LSTM have learned a recurring daily pattern?
- Could correlated observations across several hours share attribution?
- What other data might improve the forecast, such as day of week, weather, or occupancy?

---

## 22. Knowledge-check questions

1. Why is an LSTM suitable for sequential data?
2. What does the lookback window represent in this lab?
3. What is the shape of the LSTM input tensor?
4. Why do we split the time series chronologically?
5. What does the baseline represent in Integrated Gradients?
6. What does a positive Integrated Gradients score mean?
7. What does a negative Integrated Gradients score mean?
8. What is the completeness property?
9. What does Captum's convergence delta measure?
10. Does Integrated Gradients prove that an input time step caused the forecast?

### Suggested answers

1. An LSTM processes observations in sequence and maintains internal state as it reads the sequence.
2. It is the previous 24 hourly readings supplied to the model.
3. `[number_of_samples, 24, 1]`.
4. To respect time order and prevent future observations from being used to train a model evaluated on the past.
5. It is the reference input from which the attribution path begins. Here, a zero in standardised space represents training-period average consumption.
6. The input time step moves the model output upward relative to the selected baseline.
7. The input time step moves the model output downward relative to the selected baseline.
8. The sum of attributions approximates the difference between the model output at the input and at the baseline.
9. It estimates the numerical approximation error associated with the completeness property.
10. No. It explains the model's output relative to a baseline; it is not causal evidence.

---

## 23. Suggested 60–75 minute instructor flow

| Time | Demonstration |
|---|---|
| 0–5 min | Introduce the smart-energy forecasting scenario |
| 5–15 min | Download, inspect, and resample the UCI dataset |
| 15–25 min | Explain chronological splitting, scaling, and 24-hour windows |
| 25–40 min | Build and train the LSTM |
| 40–45 min | Evaluate the model and select one forecast |
| 45–55 min | Introduce Integrated Gradients and calculate attributions |
| 55–65 min | Interpret the time-step plot and verify completeness |
| 65–75 min | Compare baselines, discuss limitations, and conduct the knowledge check |

For a 60-minute session, reduce training epochs and treat the alternative-baseline experiment as an instructor-led extension.

---

## 24. Responsible interpretation and limitations

Emphasise these points:

1. **Attribution is not causation.** A high attribution does not prove that an hour caused the next-hour consumption.
2. **The baseline matters.** A different reference sequence can lead to different attributions.
3. **Attributions are model-specific.** A different LSTM, training run, or preprocessing method may produce different explanations.
4. **Time steps are correlated.** Electricity consumption at adjacent hours is not independent. The model can distribute attribution across multiple related observations.
5. **Model quality matters.** Explaining a poor forecast does not make it reliable.
6. **Data quality matters.** Missing values, resampling choices, and the selected time range can affect the model and its explanations.
7. **The data represents one household.** It should not be assumed to represent every household or region.

---

## 25. Closing explanation for students

Use this summary to conclude the demonstration:

> “An LSTM uses a sequence of observations to make a forecast. Integrated Gradients helps us look inside that prediction by assigning an attribution to each input time step. The attribution tells us how that time step contributes to the model's output relative to a chosen baseline. It does not prove a cause-and-effect relationship. The choice of baseline, the quality of the model, and the context of the data all matter when we interpret the result.”

### Transition to the next lab

In this lab, we explained a numerical forecast made from a sequence of electricity readings.

In the next lab on **Transformer and LLM Token Attribution**, we will move from time-series values to language and ask:

**“Which words or tokens contributed to the model's prediction?”**

---

## References

- UCI Machine Learning Repository – Individual Household Electric Power Consumption:  
  [https://archive.ics.uci.edu/dataset/235/individualhouseholdelectricpowerconsumption](https://archive.ics.uci.edu/dataset/235/individualhouseholdelectricpowerconsumption)
- UCI dataset DOI: [https://doi.org/10.24432/C58K54](https://doi.org/10.24432/C58K54)
- Captum – Integrated Gradients API: [https://captum.ai/api/integrated_gradients.html](https://captum.ai/api/integrated_gradients.html)
- Captum – Integrated Gradients guide: [https://captum.ai/docs/extension/integrated_gradients](https://captum.ai/docs/extension/integrated_gradients)
- PyTorch documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

**Dataset citation:** Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54. The dataset is licensed under CC BY 4.0; retain attribution when redistributing adapted copies.

**End of Lab 5**
