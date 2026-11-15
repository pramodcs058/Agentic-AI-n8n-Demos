# Lab 2 -- Grad-CAM: Visualising What a CNN Looks At

**Platform:** Google Colab\
**Runtime:** CPU (GPU optional)\
**Level:** Beginner to intermediate\
**Estimated time:** 20--25 minutes\
**Model:** Pre-trained ResNet-18 (ImageNet)\
**Input image:** Public sample photograph from the PyTorch Hub
repository\
**Libraries:** PyTorch, Torchvision, Matplotlib, Pillow, NumPy, Requests

------------------------------------------------------------------------

## 1. Lab objective

In this lab, you will use **Gradient-weighted Class Activation Mapping
(Grad-CAM)** to visualise image regions that contribute positively to a
CNN's selected class score.

In Lab 1, we treated ResNet-18 as a black box: we covered image patches
and observed how the prediction changed. Grad-CAM takes a different
approach. It uses gradients flowing back from a selected class score to
the feature maps of a convolutional layer.

The method:

1.  Loads an image and obtains the model's prediction.
2.  Selects a target class to explain.
3.  Captures feature maps from a convolutional layer.
4.  Computes gradients of the target class score with respect to those
    feature maps.
5.  Averages the gradients spatially to obtain channel weights.
6.  Combines the feature maps using those weights.
7.  Applies ReLU and resizes the resulting heatmap to the input image
    size.
8.  Overlays the heatmap on the original image.

**Core idea:** Grad-CAM highlights image regions that have a positive
influence on a chosen class score, according to the selected
convolutional layer and the model's gradients.

> Important: A Grad-CAM heatmap is a coarse, class-specific visual
> explanation. It does not prove that the model understands the object,
> nor does it show the complete internal reasoning of the network.

------------------------------------------------------------------------

## 2. Learning outcomes

By the end of this lab, students should be able to:

-   Load a pre-trained ResNet-18 model.
-   Identify a suitable convolutional layer for Grad-CAM.
-   Explain the purpose of forward hooks and gradients.
-   Calculate channel weights from gradients.
-   Generate and overlay a Grad-CAM heatmap.
-   Change the target class and compare explanations.
-   Explain the strengths and limitations of Grad-CAM.

------------------------------------------------------------------------

## 3. Classroom scenario

Imagine that a computer-vision system is used to classify photographs of
animals. It predicts that a photograph contains a dog.

A developer asks:

> "Which regions of the photograph provide evidence for the dog
> prediction? Is the model focusing on the animal, or is it responding
> to irrelevant background details?"

Grad-CAM helps us inspect the regions in the image that are important to
the selected class score.

### Opening questions

Ask students before running the notebook:

1.  Does a CNN examine every pixel equally?
2.  What do you think the final convolutional layers represent?
3.  If the model predicts "dog", should the heatmap cover the whole dog
    or only some discriminative regions?
4.  Will Grad-CAM produce the same heatmap if we select a different
    class?

Return to these questions after generating the heatmap.

------------------------------------------------------------------------

## 4. Start Google Colab

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Select **New notebook**.
3.  Rename it to `Lab-2-Grad-CAM`.
4.  Select **Runtime → Change runtime type**.
5.  Choose **CPU**. A GPU is optional.
6.  Execute the notebook cells in sequence.

The notebook downloads a public sample image and pre-trained ResNet-18
weights. An internet connection is needed for the first run.

------------------------------------------------------------------------

## 5. Step 1 -- Install and import libraries

PyTorch and Torchvision are normally available in Colab. Install the
supporting libraries.

``` python
%pip -q install matplotlib pillow requests
```

Import the packages.

``` python
import io
import requests
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
```

Check the runtime.

``` python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("Device:", device)
```

**Instructor explanation**

-   PyTorch runs the neural network and calculates gradients.
-   Torchvision provides the pre-trained ResNet-18 model.
-   Matplotlib displays the original image and Grad-CAM results.
-   Pillow opens the image.
-   NumPy helps with image and heatmap visualisation.

This lab works on CPU. A GPU may make execution faster, but it is not
required.

------------------------------------------------------------------------

## 6. Step 2 -- Download and display the sample image

For continuity with Lab 1, use the same public dog image from the
PyTorch Hub repository.

``` python
image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"

response = requests.get(image_url, timeout=30)
response.raise_for_status()

original_image = Image.open(io.BytesIO(response.content)).convert("RGB")

plt.figure(figsize=(7, 5))
plt.imshow(original_image)
plt.title("Original sample image")
plt.axis("off")
plt.show()
```

**Expected output:** A sample photograph containing a dog.

**Instructor explanation**

We are reusing the same image and model family from Lab 1 so that
students can compare two different explainability approaches:

-   Occlusion changes the input and observes the output.
-   Grad-CAM uses gradients and internal convolutional feature maps.

Ask students to predict which region the CNN may highlight.

------------------------------------------------------------------------

## 7. Step 3 -- Load the pre-trained ResNet-18 model

ResNet-18 is a convolutional neural network trained on ImageNet. It
predicts scores for 1,000 image categories.

``` python
weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)
model = model.to(device)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]

print("ResNet-18 loaded successfully.")
print("Number of ImageNet classes:", len(categories))
```

**Instructor explanation**

-   `ResNet18_Weights.DEFAULT` loads the recommended pre-trained
    weights.
-   `model.eval()` sets the model to evaluation mode.
-   `weights.transforms()` provides the preprocessing expected by the
    model.
-   We are not training or changing the model in this exercise.

**Teaching point:** Grad-CAM is a post-hoc explanation technique. It is
applied to an already trained model.

------------------------------------------------------------------------

## 8. Step 4 -- Preprocess the image

The model expects a normalised image tensor with dimensions
`(batch, channels, height, width)`.

``` python
input_tensor = preprocess(original_image)
input_batch = input_tensor.unsqueeze(0).to(device)

print("Input tensor shape:", input_batch.shape)
```

Expected shape:

``` text
torch.Size([1, 3, 224, 224])
```

The dimensions mean:

-   `1`: one image in the batch.
-   `3`: RGB channels.
-   `224`: image height.
-   `224`: image width.

Create a helper function to convert the normalised tensor back to a
displayable RGB image.

``` python
def tensor_to_display_image(tensor):
    """
    Convert a normalised ImageNet tensor into an RGB NumPy image.
    Input shape: (3, H, W)
    Output shape: (H, W, 3)
    """
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=tensor.device
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=tensor.device
    ).view(3, 1, 1)

    image = tensor.detach() * std + mean
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()

    return image


display_image = tensor_to_display_image(input_batch[0])

plt.figure(figsize=(6, 5))
plt.imshow(display_image)
plt.title("Preprocessed image")
plt.axis("off")
plt.show()
```

**Instructor explanation**

The model requires a numerical tensor, not a raw JPEG file. We use the
exact preprocessing associated with the selected ResNet-18 weights to
ensure consistency.

------------------------------------------------------------------------

## 9. Step 5 -- Get the original prediction

Run the image through the model and inspect its top-five predictions.

``` python
with torch.inference_mode():
    logits = model(input_batch)
    probabilities = torch.softmax(logits, dim=1)

top_probabilities, top_indices = torch.topk(
    probabilities[0], k=5
)

print("Top 5 predictions:\n")

for rank, (probability, class_index) in enumerate(
    zip(top_probabilities, top_indices), start=1
):
    class_name = categories[class_index.item()]
    confidence = probability.item() * 100

    print(f"{rank}. {class_name:25s} {confidence:6.2f}%")
```

Choose the highest-scoring class as the default target.

``` python
target_class = probabilities[0].argmax().item()
target_name = categories[target_class]

print("Target class:", target_name)
print(
    "Target probability:",
    f"{probabilities[0, target_class].item() * 100:.2f}%"
)
```

**Instructor explanation**

The model returns one score per ImageNet class. We apply softmax to
convert the scores into probabilities.

Grad-CAM is class-specific. We must decide which class score we want to
explain.

For this first experiment, we select the model's highest-scoring class.

------------------------------------------------------------------------

## 10. Step 6 -- Understand the Grad-CAM architecture

Grad-CAM needs two pieces of information from a convolutional layer:

1.  **Feature maps (activations):** What patterns or spatial information
    did the layer produce?
2.  **Gradients:** How does the selected class score change with respect
    to those feature maps?

For ResNet-18, we will use:

``` python
target_layer = model.layer4[-1]
```

This is the final residual block in the final convolutional stage.

We use this layer because it contains high-level spatial features that
are useful for class-specific localisation.

### Grad-CAM formula

Let:

-   (A\^k) represent the feature map for channel (k).
-   (y\^c) represent the score for target class (c).
-   \(Z\) represent the number of spatial positions in the feature map.

The channel importance weight is:

\[ `\alpha`{=tex}\_k\^c = `\frac{1}{Z}`{=tex} `\sum`{=tex}\_i
`\sum`{=tex}\_j `\frac{\partial y^c}{\partial A_{ij}^k}`{=tex} \]

The Grad-CAM heatmap is:

\[ L\_{`\mathrm{Grad-CAM}`{=tex}}\^c = `\operatorname{ReLU}`{=tex}
`\left`{=tex}( `\sum`{=tex}\_k `\alpha`{=tex}\_k\^c A\^k `\right`{=tex})
\]

**Explain the formula in simple terms**

-   Calculate how strongly the target class score depends on each
    feature-map location.
-   Average the gradients spatially to get one weight per channel.
-   Multiply each feature map by its weight.
-   Add the weighted feature maps together.
-   Apply ReLU to retain positive evidence.

The resulting map is then resized to the original image dimensions.

------------------------------------------------------------------------

## 11. Step 7 -- Register a forward hook

A forward hook lets us capture the output of a selected layer when the
model processes an image.

``` python
activations = {}

def forward_hook(module, inputs, output):
    activations["features"] = output

target_layer = model.layer4[-1]

hook_handle = target_layer.register_forward_hook(forward_hook)

print("Forward hook registered.")
```

**Instructor explanation**

A hook is a function that runs when a particular layer executes.

Here:

-   `target_layer` identifies the final residual block.
-   `register_forward_hook()` captures that layer's output.
-   The captured output contains the feature maps needed by Grad-CAM.

Do not confuse a forward hook with a backward hook. The forward hook
captures activations; we will capture gradients separately.

------------------------------------------------------------------------

## 12. Step 8 -- Calculate gradients for the target class

We now perform a forward pass and calculate the gradient of the selected
class score with respect to the captured feature maps.

``` python
model.zero_grad(set_to_none=True)
activations.clear()

# Forward pass
logits = model(input_batch)

# Capture the selected layer's feature maps
feature_maps = activations["features"]

print("Feature map shape:", feature_maps.shape)
```

For ResNet-18, the expected shape is:

``` text
torch.Size([1, 512, 7, 7])
```

The dimensions mean:

-   `1`: batch size.
-   `512`: feature-map channels.
-   `7 × 7`: spatial dimensions.

Now select the target class score and register a gradient hook on the
captured feature maps.

``` python
gradients = {}

def save_gradient(gradient):
    gradients["value"] = gradient

feature_maps.register_hook(save_gradient)

# Score for the class being explained
target_score = logits[0, target_class]

# Backpropagate the selected class score
target_score.backward()

print("Gradient shape:", gradients["value"].shape)
```

**Expected output:** The gradient tensor should have the same dimensions
as the feature maps.

**Instructor explanation**

The important line is:

``` python
target_score.backward()
```

This calculates gradients for the selected class score.

The gradient hook stores the gradient flowing back to the feature maps.

We now have both required ingredients:

-   `feature_maps`: activations from the selected convolutional layer.
-   `gradients["value"]`: gradients of the target class score with
    respect to those activations.

**Teaching point:** Grad-CAM uses gradients to determine which
feature-map channels are important for the selected class.

------------------------------------------------------------------------

## 13. Step 9 -- Calculate Grad-CAM channel weights

Average each feature-map channel's gradients across its height and
width.

``` python
feature_gradients = gradients["value"]

weights_per_channel = feature_gradients.mean(
    dim=(2, 3),
    keepdim=True
)

print("Channel weights shape:", weights_per_channel.shape)
```

Expected shape:

``` text
torch.Size([1, 512, 1, 1])
```

**Instructor explanation**

Each of the 512 channels receives one importance weight.

The average is calculated across the spatial dimensions, leaving one
value per channel.

This implements:

\[ `\alpha`{=tex}\_k\^c = `\operatorname{GlobalAveragePool}`{=tex}
`\left`{=tex}( `\frac{\partial y^c}{\partial A^k}`{=tex} `\right`{=tex})
\]

------------------------------------------------------------------------

## 14. Step 10 -- Generate the Grad-CAM heatmap

Multiply the feature maps by their corresponding weights, sum across
channels and apply ReLU.

``` python
weighted_features = weights_per_channel * feature_maps

# Sum across channels
cam = weighted_features.sum(dim=1, keepdim=True)

# Keep only positive evidence
cam = F.relu(cam)

print("Raw Grad-CAM shape:", cam.shape)
```

Expected shape:

``` text
torch.Size([1, 1, 7, 7])
```

Resize the heatmap to the model input size.

``` python
cam_resized = F.interpolate(
    cam,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

cam_resized = cam_resized[0, 0]

# Normalise the heatmap for display
cam_min = cam_resized.min()
cam_max = cam_resized.max()

if (cam_max - cam_min).item() > 1e-8:
    cam_resized = (
        (cam_resized - cam_min) /
        (cam_max - cam_min)
    )
else:
    cam_resized = torch.zeros_like(cam_resized)

gradcam_heatmap = cam_resized.detach().cpu().numpy()

print("Grad-CAM heatmap generated.")
print("Heatmap dimensions:", gradcam_heatmap.shape)
```

**Instructor explanation**

-   Weighted feature maps combine spatial information with class
    importance.
-   ReLU removes negative contributions, retaining regions that provide
    positive evidence for the target class.
-   Interpolation expands the coarse 7 × 7 map to 224 × 224.
-   Normalisation scales the values between 0 and 1 for visualisation.

The 224 × 224 heatmap is an upsampled visualisation. It does not mean
the CNN originally produced a 224 × 224 explanation.

------------------------------------------------------------------------

## 15. Step 11 -- Display the Grad-CAM heatmap

Display the original image, the heatmap and the overlay side by side.

``` python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original image
axes[0].imshow(display_image)
axes[0].set_title("Original image")
axes[0].axis("off")

# Grad-CAM heatmap
im = axes[1].imshow(
    gradcam_heatmap,
    cmap="jet",
    vmin=0,
    vmax=1
)
axes[1].set_title(f"Grad-CAM: {target_name}")
axes[1].axis("off")

fig.colorbar(
    im,
    ax=axes[1],
    fraction=0.046,
    pad=0.04,
    label="Relative activation"
)

# Overlay
axes[2].imshow(display_image)
axes[2].imshow(
    gradcam_heatmap,
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)
axes[2].set_title("Grad-CAM overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()
```

**Expected output**

Three visualisations:

1.  Original photograph.
2.  Grad-CAM heatmap.
3.  Heatmap overlaid on the photograph.

Warm regions indicate stronger positive Grad-CAM responses for the
selected class.

The exact highlighted region depends on the model, target class and
selected layer.

### Ask students

-   Does the heatmap highlight the dog?
-   Is the entire dog highlighted or only some regions?
-   Does the background receive any activation?
-   What would it mean if the model highlighted an irrelevant object?

**Important:** A heatmap is evidence about the model's behaviour, not
proof that the prediction is correct.

------------------------------------------------------------------------

## 16. Step 12 -- Explain a different class

Grad-CAM is class-specific. We can select another class from the
top-five predictions and generate a different heatmap.

Display the predictions again.

``` python
with torch.inference_mode():
    probabilities = torch.softmax(model(input_batch), dim=1)

top_probabilities, top_indices = torch.topk(
    probabilities[0], k=5
)

for rank, (probability, class_index) in enumerate(
    zip(top_probabilities, top_indices), start=1
):
    print(
        rank,
        categories[class_index.item()],
        f"{probability.item() * 100:.2f}%"
    )
```

Select the second-highest prediction.

``` python
alternative_class = top_indices[1].item()
alternative_name = categories[alternative_class]

print("Alternative class:", alternative_name)
```

Create a reusable function to calculate Grad-CAM for any selected class.

``` python
def generate_gradcam(model, image_tensor, target_layer, target_class):
    """
    Generate a Grad-CAM heatmap for one image and one target class.

    Args:
        model: Evaluated classification model.
        image_tensor: Input tensor of shape (1, 3, H, W).
        target_layer: Convolutional layer to explain.
        target_class: ImageNet class index.

    Returns:
        Normalised heatmap as a 2D NumPy array.
    """

    model.eval()

    captured = {}

    def forward_hook(module, inputs, output):
        captured["features"] = output

        # Save the gradient flowing into this feature-map tensor.
        output.register_hook(
            lambda grad: captured.update({"gradients": grad})
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        logits = model(image_tensor)
        score = logits[0, target_class]

        score.backward()

        feature_maps = captured["features"]
        feature_gradients = captured["gradients"]

        channel_weights = feature_gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = F.relu(
            (channel_weights * feature_maps).sum(
                dim=1,
                keepdim=True
            )
        )

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )[0, 0]

        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min).item() > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
```

Generate the alternative heatmap.

``` python
alternative_heatmap = generate_gradcam(
    model=model,
    image_tensor=input_batch,
    target_layer=target_layer,
    target_class=alternative_class
)

plt.figure(figsize=(7, 5))
plt.imshow(display_image)
plt.imshow(
    alternative_heatmap,
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)
plt.title(f"Grad-CAM for: {alternative_name}")
plt.axis("off")
plt.show()
```

**Instructor explanation**

The input image and network remain the same. Only the target class
changes.

Grad-CAM therefore answers a class-specific question:

-   Which regions provide positive evidence for class A?
-   Which regions provide positive evidence for class B?

The heatmaps may be different because the gradients are calculated with
respect to different class scores.

------------------------------------------------------------------------

## 17. Step 13 -- Compare Occlusion and Grad-CAM

This is an important teaching activity because students have now seen
two explainability techniques applied to the same image and model.

  -----------------------------------------------------------------------
  Aspect                  Occlusion               Grad-CAM
  ----------------------- ----------------------- -----------------------
  Main idea               Hide image patches and  Use gradients and
                          observe score changes   convolutional feature
                                                  maps

  Model access            Input and output        Internal convolutional
                          predictions             layer and gradients

  Gradients required      No                      Yes

  Model-agnostic          Yes, in the             No, model-specific
                          input-output sense      

  Output                  Patch-based sensitivity Class activation
                          map                     heatmap

  Computational cost      Many forward passes     Forward and backward
                                                  pass

  Main limitation         Slow; patch and fill    Coarse; layer and
                          dependent               gradient dependent
  -----------------------------------------------------------------------

### Generate a side-by-side comparison

If the Lab 1 notebook has already been executed in the same Colab
runtime, its `heatmap` variable may still exist. Otherwise, run the
following cell to calculate an occlusion map again.

``` python
def calculate_occlusion_map(model, image_tensor, target_class,
                             patch_size=32, stride=16):
    """
    A compact occlusion implementation for comparison with Grad-CAM.
    The grey replacement is correctly converted to ImageNet-normalised values.
    """

    model.eval()

    _, _, height, width = image_tensor.shape

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=image_tensor.device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=image_tensor.device
    ).view(1, 3, 1, 1)

    grey = (0.5 - mean) / std

    with torch.inference_mode():
        base_probability = torch.softmax(
            model(image_tensor), dim=1
        )[0, target_class].item()

    rows = list(range(0, height - patch_size + 1, stride))
    cols = list(range(0, width - patch_size + 1, stride))

    if rows[-1] != height - patch_size:
        rows.append(height - patch_size)

    if cols[-1] != width - patch_size:
        cols.append(width - patch_size)

    result = np.zeros((len(rows), len(cols)), dtype=np.float32)

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            modified = image_tensor.clone()

            modified[
                :, :, row:row + patch_size,
                col:col + patch_size
            ] = grey

            with torch.inference_mode():
                probability = torch.softmax(
                    model(modified), dim=1
                )[0, target_class].item()

            result[i, j] = base_probability - probability

    return result
```

Calculate the occlusion heatmap for the same target class.

``` python
occlusion_heatmap = calculate_occlusion_map(
    model,
    input_batch,
    target_class,
    patch_size=32,
    stride=16
)

occlusion_tensor = torch.tensor(
    occlusion_heatmap,
    dtype=torch.float32
).view(1, 1, *occlusion_heatmap.shape)

occlusion_resized = F.interpolate(
    occlusion_tensor,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)[0, 0].numpy()
```

Display both methods.

``` python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].imshow(display_image)
axes[0].imshow(
    occlusion_resized,
    cmap="hot",
    alpha=0.5
)
axes[0].set_title("Occlusion sensitivity")
axes[0].axis("off")

axes[1].imshow(display_image)
axes[1].imshow(
    gradcam_heatmap,
    cmap="jet",
    alpha=0.5,
    vmin=0,
    vmax=1
)
axes[1].set_title("Grad-CAM")
axes[1].axis("off")

plt.tight_layout()
plt.show()
```

**Discussion**

-   Do both techniques highlight exactly the same regions?
-   Why might their results differ?
-   Which technique requires access to model internals?
-   Which technique would you choose if the model did not expose
    gradients?

**Instructor explanation**

The techniques measure different things. Occlusion measures how the
prediction changes after masking an input region. Grad-CAM uses
gradients to construct a class-specific map from an internal feature
layer. Their outputs should not be expected to match exactly.

------------------------------------------------------------------------

## 18. Step 14 -- Student experimentation tasks

### Activity A -- Change the target class

Generate Grad-CAM heatmaps for the top two or three predicted classes.

Record:

  Target class        Highlighted region   What did you observe?
  ------------------- -------------------- -----------------------
  Top prediction                           
  Second prediction                        
  Third prediction                         

### Activity B -- Change the convolutional layer

Try a different convolutional stage.

``` python
# Compare with an earlier convolutional stage
earlier_layer = model.layer3[-1]

earlier_heatmap = generate_gradcam(
    model,
    input_batch,
    earlier_layer,
    target_class
)

plt.figure(figsize=(7, 5))
plt.imshow(display_image)
plt.imshow(
    earlier_heatmap,
    cmap="jet",
    alpha=0.45,
    vmin=0,
    vmax=1
)
plt.title("Grad-CAM using layer3[-1]")
plt.axis("off")
plt.show()
```

Compare it with the original Grad-CAM from `model.layer4[-1]`.

**Ask students:** Does the earlier layer produce a more detailed or less
class-specific visualisation?

Earlier layers often preserve more spatial detail, while deeper layers
generally contain more class-specific information. The exact visual
result depends on the image and model.

### Activity C -- Use another photograph

Replace the sample image with a publicly accessible photograph of a cat,
bird, vehicle or another ImageNet object.

``` python
image_url = "PASTE_ANOTHER_PUBLIC_IMAGE_URL_HERE"
```

Download the image, preprocess it using `preprocess`, obtain the target
class and run `generate_gradcam()` again.

**Important:** The model is trained on ImageNet. Use a clear image of an
object that is likely to belong to one of its 1,000 classes.

------------------------------------------------------------------------

## 19. Troubleshooting

  -----------------------------------------------------------------------------
  Problem                   Possible reason         Solution
  ------------------------- ----------------------- ---------------------------
  Model download fails      Internet connection     Retry the model-loading
                            issue                   cell

  Image download fails      URL or network issue    Retry or use another public
                                                    image

  `KeyError: 'features'`    Forward hook did not    Check the selected layer
                            execute                 and run a forward pass

  `KeyError: 'gradients'`   Gradient hook was not   Ensure the forward pass is
                            triggered               not inside
                                                    `torch.no_grad()` or
                                                    `torch.inference_mode()`

  Gradient is missing       Backward pass was not   Run
                            executed                `target_score.backward()`

  Heatmap is blank          ReLU removed all        Check target class and
                            positive values or map  selected layer
                            is constant             

  Heatmap is coarse         Final feature map has   Try an earlier
                            low spatial resolution  convolutional layer

  Wrong class is explained  Incorrect target class  Inspect top-five
                            index                   predictions and set the
                                                    target explicitly

  Runtime is slow           CPU execution or model  Allow the initial download
                            download                to finish; use a smaller
                                                    number of images
  -----------------------------------------------------------------------------

### Important implementation notes

-   Do not wrap the Grad-CAM forward pass in `torch.inference_mode()` or
    `torch.no_grad()`. Gradients are required.
-   Use `model.eval()` to disable training-specific behaviour.
-   Always remove temporary hooks after use to avoid accumulating hooks
    across repeated experiments.
-   Use the class index from the same ImageNet category list returned by
    the selected weights.

------------------------------------------------------------------------

## 20. Limitations of Grad-CAM

Discuss these limitations with students:

1.  **Coarse resolution:** The final ResNet-18 feature map is only 7
    × 7. Upsampling makes it easier to view but does not create
    additional detail.
2.  **Layer dependency:** Different convolutional layers can produce
    different heatmaps.
3.  **Class dependency:** The heatmap changes with the target class.
4.  **Gradient dependency:** Gradient behaviour can affect the
    explanation.
5.  **Positive evidence only:** ReLU retains positive contributions and
    removes negative contributions from the visualisation.
6.  **Not proof of correctness:** A plausible heatmap does not guarantee
    that the prediction is correct or unbiased.
7.  **Model-specific:** Grad-CAM needs access to a suitable
    convolutional layer and gradients, so it is not a fully
    model-agnostic method.

### Important distinction

Grad-CAM is:

-   Post-hoc: it explains a model after training.
-   Local: it explains a particular image prediction.
-   Model-specific: it uses the internal feature maps and gradients of a
    CNN.

------------------------------------------------------------------------

## 21. Final discussion and knowledge check

Ask students to answer without looking at the code.

1.  What does Grad-CAM stand for?
2.  Why do we select a convolutional layer?
3.  What information does the forward hook capture?
4.  What information do gradients provide?
5.  Why do we average gradients across height and width?
6.  Why is ReLU applied?
7.  Why is the final heatmap coarse?
8.  Why is Grad-CAM model-specific?
9.  Does a highlighted region prove that the model is correct?
10. How is Grad-CAM different from occlusion?

### Suggested answers

  -----------------------------------------------------------------------
  Question                            Expected answer
  ----------------------------------- -----------------------------------
  Grad-CAM?                           Gradient-weighted Class Activation
                                      Mapping

  Why a convolutional layer?          It contains spatial feature maps

  Forward hook?                       Captures the selected layer's
                                      activations

  Gradients?                          Measure sensitivity of the target
                                      score to feature maps

  Average gradients?                  Produces one importance weight per
                                      channel

  Why ReLU?                           Retains positive evidence for the
                                      target class

  Why coarse?                         The convolutional feature map has
                                      low spatial resolution

  Why model-specific?                 It uses internal layers and
                                      gradients

  Does it prove correctness?          No, it is an approximate visual
                                      explanation

  Difference from occlusion?          Occlusion masks input patches;
                                      Grad-CAM uses internal feature maps
                                      and gradients
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 22. Lab conclusion

In this lab, we used a pre-trained ResNet-18 model and Grad-CAM to
investigate which image regions contribute positively to a selected
class score.

We captured feature maps from the final convolutional stage, calculated
gradients, derived channel weights and generated a heatmap over the
original image.

### Key takeaway

**Occlusion asks what happens when we hide part of an image. Grad-CAM
uses the CNN's gradients and feature maps to highlight regions that
support a selected class prediction.**

### Link to the next lab -- DeepSHAP and DeepLIFT

So far, we have focused on image-based explanations.

In Lab 3, we will move to tabular data and investigate how individual
features contribute positively or negatively to a model's prediction.

Ask students:

> "If an AI system rejects an education loan, can we identify which
> applicant features pushed the prediction towards rejection?"

This introduces feature attribution methods such as SHAP and DeepLIFT.

------------------------------------------------------------------------

## References

-   [Google Colab](https://colab.research.google.com/)
-   [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
-   [Torchvision ResNet-18
    documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
-   [PyTorch Hub sample dog
    image](https://github.com/pytorch/hub/blob/master/images/dog.jpg)
-   [Grad-CAM paper -- Selvaraju et
    al. (2017)](https://arxiv.org/abs/1610.02391)
-   [Matplotlib documentation](https://matplotlib.org/stable/index.html)
