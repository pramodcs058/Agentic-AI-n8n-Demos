# Lab 1 -- Occlusion Sensitivity for Explainable AI (XAI)

**Platform:** Google Colab\
**Runtime:** CPU (GPU is optional, not required)\
**Level:** Beginner\
**Estimated time:** 20--25 minutes\
**Model:** Pre-trained ResNet-18 (ImageNet)\
**Input:** Public dog photograph\
**Libraries:** PyTorch, Torchvision, Matplotlib, Pillow, NumPy

------------------------------------------------------------------------

## 1. Lab objective

In this lab, you will use **occlusion sensitivity** to investigate which
regions of an image influence a neural network's prediction.

A pre-trained image-classification model may predict that an image
contains a dog. But how can we investigate whether the model is using
the dog itself, the background, or some other region?

Occlusion provides a simple experiment:

1.  Classify the original image and record the probability of the
    predicted class.
2.  Cover a small image patch with a neutral grey region.
3.  Classify the modified image again.
4.  Measure how much the original class probability changes.
5.  Move the patch across the image and repeat.
6.  Visualise the score changes as an occlusion heatmap.

**Core idea:** If hiding a region causes the target-class probability to
drop substantially, that region was important to the model's prediction
in this experiment.

> Important: Occlusion measures sensitivity to masking. It does not
> reveal the model's complete internal reasoning or prove that the
> highlighted region is causally responsible in the real world.

## 2. Learning outcomes

By the end of this lab, students should be able to:

-   Load and use a pre-trained CNN model.
-   Obtain a classification prediction and confidence score.
-   Explain how patch-based occlusion works.
-   Calculate the drop in a class probability after masking image
    regions.
-   Interpret an occlusion heatmap.
-   Explain why patch size, stride and replacement colour affect the
    result.

## 3. Scenario for the classroom

Imagine that a wildlife-monitoring application uses an AI model to
identify animals from camera images.

The model predicts **Golden Retriever** for a photograph. A wildlife
researcher asks:

> "Which parts of this photograph influenced the model's prediction? Did
> it focus on the dog, or was it influenced by the background?"

We will investigate this by covering different parts of the photograph
and observing how the prediction changes.

### Opening questions for students

Ask the class before running the code:

1.  If we cover the dog's face, what do you think will happen to the
    prediction score?
2.  If we cover only the sky or background, will the score change
    significantly?
3.  Will covering a larger area always give us a better explanation?

Do not reveal the answers immediately. Return to these questions after
viewing the heatmap.

## 4. Start Google Colab

1.  Open [Google Colab](https://colab.research.google.com/).
2.  Select **New notebook**.
3.  Rename the notebook to `Lab-1-Occlusion`.
4.  Select **Runtime → Change runtime type**.
5.  Choose **CPU**. A GPU is optional.
6.  Run the cells in the order given below.

The notebook downloads a small public sample image and the pre-trained
ResNet-18 weights. An internet connection is therefore required for the
first run.

## 5. Step 1 -- Install and import libraries

Torch and Torchvision are generally available in a standard Colab
runtime. Install only the supporting visualisation and image libraries
if needed.

``` python
%pip -q install matplotlib pillow requests
```

Import the required packages.

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

-   `torch` provides tensor operations and neural-network execution.
-   `torchvision` provides pre-trained computer-vision models and image
    transformations.
-   `matplotlib` displays the original image and heatmaps.
-   `PIL` opens the downloaded image.
-   We use CPU-compatible code, so students do not need a paid Colab
    plan or GPU.

## 6. Step 2 -- Download and display the sample image

We will use a publicly accessible sample dog image hosted by the PyTorch
Hub repository.

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

**Expected output:** A photograph of a dog.

**Instructor explanation**

This is the input that will be supplied to the neural network. We will
keep the original image unchanged and create modified copies for the
occlusion experiment.

Ask students:

-   What objects can you identify?
-   Which region do you think the model will use?
-   Could the background influence the classification?

## 7. Step 3 -- Load the pre-trained ResNet-18 model

ResNet-18 is a convolutional neural network trained on the ImageNet
dataset. It can classify images into 1,000 ImageNet categories.

``` python
weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)
model = model.to(device)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]

print("Model loaded successfully.")
print("Number of ImageNet classes:", len(categories))
```

**Instructor explanation**

-   ResNet-18 is already trained; we do not train it in this lab.
-   `weights=ResNet18_Weights.DEFAULT` loads the recommended pre-trained
    weights.
-   `model.eval()` switches the network into evaluation mode.
-   `weights.transforms()` supplies the preprocessing expected by these
    weights, including resizing, cropping and normalisation.
-   The model returns scores for 1,000 ImageNet categories.

**Teaching point:** We are explaining the behaviour of an existing
trained model. Occlusion does not modify or retrain the model.

## 8. Step 4 -- Preprocess the image

The model expects an image tensor with the correct dimensions and
normalisation.

``` python
input_tensor = preprocess(original_image)
input_batch = input_tensor.unsqueeze(0).to(device)

print("Input tensor shape:", input_batch.shape)
```

Expected shape:

``` text
Input tensor shape: torch.Size([1, 3, 224, 224])
```

The dimensions represent:

-   `1`: one image in the batch.
-   `3`: RGB colour channels.
-   `224`: image height.
-   `224`: image width.

Create a helper function to display the preprocessed image in its normal
RGB colour range.

``` python
def tensor_to_display_image(tensor):
    """
    Convert a normalised image tensor into a NumPy RGB image
    suitable for Matplotlib display.
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


plt.figure(figsize=(6, 5))
plt.imshow(tensor_to_display_image(input_batch[0]))
plt.title("Image after model preprocessing")
plt.axis("off")
plt.show()
```

**Instructor explanation**

Neural networks expect numerical tensors, not raw image files.
Preprocessing converts the image into the format expected by the model.

The normalisation values are part of the ResNet-18 ImageNet
preprocessing. We must respect this when masking pixels later.

## 9. Step 5 -- Get the original prediction

Run the original image through ResNet-18.

``` python
with torch.inference_mode():
    logits = model(input_batch)
    probabilities = torch.softmax(logits, dim=1)

top_probabilities, top_indices = torch.topk(
    probabilities[0], k=5
)

print("Top 5 predictions:\n")

for probability, class_index in zip(top_probabilities, top_indices):
    class_name = categories[class_index.item()]
    confidence = probability.item() * 100

    print(f"{class_name:25s} {confidence:6.2f}%")
```

Identify the highest-scoring class. This will be our target class for
the occlusion experiment.

``` python
target_class = probabilities[0].argmax().item()
target_name = categories[target_class]
base_score = probabilities[0, target_class].item()

print("Target class:", target_name)
print(f"Original target probability: {base_score:.4f}")
print(f"Original target confidence: {base_score * 100:.2f}%")
```

**Expected output**

The exact scores may vary with model/library versions, but the
prediction should identify a dog-related ImageNet class.

**Instructor explanation**

The model produces a probability distribution over its 1,000 categories.
We select the highest-scoring category as the target.

For the rest of the lab, we ask one consistent question:

> How does hiding each image region affect the probability of this
> original target class?

We keep the target class fixed. We do not choose a new target class
after each patch is hidden.

## 10. Step 6 -- Understand the occlusion parameters

We will slide a square grey patch across the image.

  ------------------------------------------------------------------------
  Parameter                                    Value Meaning
  --------------------- ---------------------------- ---------------------
  Patch size                                 32 × 32 Number of pixels
                                                     covered in each test

  Stride                                          16 Number of pixels the
                                                     patch moves each time

  Fill colour                         Grey (RGB 0.5) Replacement value for
                                                     the covered region

  Target class              Original predicted class Class whose
                                                     probability is
                                                     monitored
  ------------------------------------------------------------------------

With a 224 × 224 image, a patch size of 32 and stride of 16 produce:

\[ N =
`\left`{=tex}`\lfloor`{=tex}`\frac{224-32}{16}`{=tex}`\right`{=tex}`\rfloor `{=tex}+
1 = 13 \]

There are 13 positions vertically and 13 horizontally, resulting in:

\[ 13 `\times 13`{=tex} = 169 `\text{ occlusion tests}`{=tex} \]

**Instructor explanation**

-   A smaller patch gives finer spatial detail but requires more model
    evaluations.
-   A larger patch gives coarser explanations and may hide several
    useful features together.
-   A smaller stride creates more overlapping tests and takes longer.
-   A larger stride is faster but may miss small important regions.

## 11. Step 7 -- Implement the occlusion function

The following function performs the experiment.

For every patch position, it:

1.  Creates a copy of the original preprocessed image.
2.  Replaces the selected patch with grey.
3.  Runs the modified image through ResNet-18.
4.  Gets the probability of the original target class.
5.  Calculates the score drop.

We use a grey value of 0.5 in the original RGB image range. Because the
image has already been normalised, we convert that grey value into the
corresponding normalised RGB tensor values before applying the patch.

``` python
def occlusion_map(model, image_tensor, target_class,
                  patch_size=32, stride=16,
                  grey_value=0.5):
    """
    Calculate an occlusion sensitivity map.

    Args:
        model: Trained image-classification model.
        image_tensor: Normalised image tensor of shape (1, 3, H, W).
        target_class: Fixed class index to investigate.
        patch_size: Width and height of the square occlusion patch.
        stride: Number of pixels moved between patch positions.
        grey_value: Replacement grey value in the original [0, 1] RGB range.

    Returns:
        score_drops: 2D NumPy array containing the probability drops.
        base_probability: Original target-class probability.
    """

    model.eval()

    _, channels, height, width = image_tensor.shape

    if patch_size > height or patch_size > width:
        raise ValueError("Patch size must not exceed image dimensions.")

    if stride <= 0:
        raise ValueError("Stride must be greater than zero.")

    # Original target-class probability
    with torch.inference_mode():
        original_logits = model(image_tensor)
        original_probability = torch.softmax(
            original_logits, dim=1
        )[0, target_class].item()

    # ImageNet normalisation parameters for RGB channels
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=image_tensor.device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=image_tensor.device
    ).view(1, 3, 1, 1)

    # Convert grey RGB value into normalised tensor values
    grey_normalised = (grey_value - mean) / std

    row_positions = list(
        range(0, height - patch_size + 1, stride)
    )
    col_positions = list(
        range(0, width - patch_size + 1, stride)
    )

    # Ensure the final edge is also tested
    if row_positions[-1] != height - patch_size:
        row_positions.append(height - patch_size)

    if col_positions[-1] != width - patch_size:
        col_positions.append(width - patch_size)

    score_drops = np.zeros(
        (len(row_positions), len(col_positions)),
        dtype=np.float32
    )

    for row_index, row in enumerate(row_positions):
        for col_index, col in enumerate(col_positions):

            # Make a fresh copy of the original image
            occluded_image = image_tensor.clone()

            # Replace the selected patch with normalised grey
            occluded_image[
                :, :, row:row + patch_size,
                col:col + patch_size
            ] = grey_normalised

            # Predict the target-class probability
            with torch.inference_mode():
                logits = model(occluded_image)
                probability = torch.softmax(
                    logits, dim=1
                )[0, target_class].item()

            # Positive drop means masking reduced the probability
            score_drops[row_index, col_index] = (
                original_probability - probability
            )

    return score_drops, original_probability
```

**Instructor explanation: Understanding the important line**

``` python
score_drop = original_probability - occluded_probability
```

For example:

-   Original dog probability = 0.94
-   Probability after hiding a region = 0.18
-   Score drop = 0.76

A large positive drop indicates that hiding this region substantially
reduced the model's confidence in the target class.

A small drop indicates that masking this patch had little effect on the
target-class probability in this particular test.

A negative value means the target-class probability increased after
masking.

## 12. Step 8 -- Generate the occlusion heatmap

Execute the function.

``` python
heatmap, original_probability = occlusion_map(
    model=model,
    image_tensor=input_batch,
    target_class=target_class,
    patch_size=32,
    stride=16,
    grey_value=0.5
)

print("Occlusion map shape:", heatmap.shape)
print("Original target probability:", original_probability)
print("Maximum score drop:", heatmap.max())
print("Minimum score drop:", heatmap.min())
```

Display the score-drop matrix.

``` python
plt.figure(figsize=(7, 5))

plt.imshow(
    heatmap,
    cmap="hot",
    interpolation="nearest"
)

plt.colorbar(label="Target probability drop")
plt.title("Occlusion score-drop matrix")
plt.xlabel("Horizontal patch position")
plt.ylabel("Vertical patch position")

plt.show()
```

**How to interpret the matrix**

-   Larger positive values: hiding that patch reduced the target
    probability substantially.
-   Values close to zero: hiding that patch had little effect.
-   Negative values: hiding that patch increased the target probability.

The matrix contains one score-drop value for each patch position. It is
not yet a pixel-level explanation.

## 13. Step 9 -- Overlay the heatmap on the original image

We will resize the coarse score-drop matrix to the image dimensions so
that students can see the important regions more easily.

``` python
heat_tensor = torch.tensor(
    heatmap,
    dtype=torch.float32
).view(1, 1, *heatmap.shape)

heatmap_resized = F.interpolate(
    heat_tensor,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)[0, 0].numpy()

display_image = tensor_to_display_image(input_batch[0])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original image
axes[0].imshow(display_image)
axes[0].set_title("Original image")
axes[0].axis("off")

# Occlusion heatmap
im = axes[1].imshow(
    heatmap_resized,
    cmap="hot",
    interpolation="bilinear"
)
axes[1].set_title("Occlusion heatmap")
axes[1].axis("off")

fig.colorbar(
    im,
    ax=axes[1],
    fraction=0.046,
    pad=0.04,
    label="Probability drop"
)

# Heatmap overlay
axes[2].imshow(display_image)
axes[2].imshow(
    heatmap_resized,
    cmap="hot",
    alpha=0.45,
    interpolation="bilinear"
)
axes[2].set_title("Original image + occlusion overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()
```

**Expected output**

Three visualisations:

1.  Original dog photograph.
2.  Occlusion sensitivity heatmap.
3.  Original photograph with the heatmap overlay.

The warm regions indicate patch locations where hiding the image caused
a larger drop in the original target-class probability.

The precise highlighted regions depend on the model prediction, patch
size, stride and image.

**Ask students**

-   Does the heatmap highlight the dog's face or body?
-   Does it highlight the background?
-   What happens when a region with a strong response is hidden?
-   Does this prove the model understands what a dog is?

## 14. Step 10 -- Compare different patch sizes

The patch size affects the explanation. Let us compare two different
configurations.

``` python
heatmap_small, _ = occlusion_map(
    model,
    input_batch,
    target_class,
    patch_size=16,
    stride=16
)

heatmap_large, _ = occlusion_map(
    model,
    input_batch,
    target_class,
    patch_size=48,
    stride=24
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(heatmap_small, cmap="hot")
axes[0].set_title("Patch = 16 × 16")
axes[0].set_xlabel("Horizontal position")
axes[0].set_ylabel("Vertical position")

axes[1].imshow(heatmap_large, cmap="hot")
axes[1].set_title("Patch = 48 × 48")
axes[1].set_xlabel("Horizontal position")
axes[1].set_ylabel("Vertical position")

plt.tight_layout()
plt.show()
```

**Instructor explanation**

The smaller patch tests more local regions. The larger patch hides more
image content at once and can produce a coarser explanation.

Ask students:

> Which patch size gives a more useful explanation for identifying a
> dog's face? Why?

There is no universally best patch size. The right choice depends on the
size of the object or feature being investigated.

## 15. Step 11 -- Classroom experiment: Change the target class

The original experiment used the model's top predicted class. We can
also investigate another class from the model's top-five predictions.

First, display the top-five predictions again.

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

Choose a class index from the displayed top-five results.

``` python
# Change this index to another class from the top-five list.
alternative_class = top_indices[1].item()

alternative_name = categories[alternative_class]

print("Alternative target class:", alternative_name)
```

Generate an occlusion map for this alternative class.

``` python
alternative_heatmap, alternative_base = occlusion_map(
    model,
    input_batch,
    alternative_class,
    patch_size=32,
    stride=16
)

plt.figure(figsize=(7, 5))
plt.imshow(alternative_heatmap, cmap="hot")
plt.colorbar(label="Probability drop")
plt.title(f"Occlusion map for: {alternative_name}")
plt.axis("off")
plt.show()
```

**Teaching point**

The same photograph can have different occlusion maps for different
target classes. Explanations are class-specific.

## 16. Step 12 -- Student activities

Ask students to complete the following experiments.

### Activity A -- Change the patch size

Run the experiment with:

-   16 × 16 patch
-   32 × 32 patch
-   48 × 48 patch

Record the following:

  Patch size     Number of positions What changed in the heatmap?
  ------------ --------------------- ------------------------------
  16 × 16        Calculate from code 
  32 × 32        Calculate from code 
  48 × 48        Calculate from code 

### Activity B -- Change the stride

Try stride values of 8, 16 and 32.

Discuss:

1.  Which configuration requires the most model evaluations?
2.  Which configuration produces a more detailed heatmap?
3.  Which configuration is faster?

### Activity C -- Use another photograph

Replace the dog image URL with another publicly accessible image, such
as a cat, bird or vehicle.

For example:

``` python
image_url = "PASTE_ANOTHER_PUBLIC_IMAGE_URL_HERE"
```

Download the new image, preprocess it, obtain the model's target
prediction and regenerate the occlusion map.

**Note:** The model is trained on ImageNet. Use a clear image of an
object likely to belong to one of its 1,000 categories.

## 17. Troubleshooting

  -----------------------------------------------------------------------
  Problem                 Possible reason         Solution
  ----------------------- ----------------------- -----------------------
  Image download fails    Network or URL issue    Retry or use another
                                                  public image URL

  Model download fails    Internet connectivity   Retry the model-loading
                                                  cell

  CUDA error              Runtime/device mismatch Restart the runtime and
                                                  use CPU

  Heatmap is mostly dark  Small score changes or  Inspect the numeric
                          colour scaling          score-drop matrix

  Heatmap is coarse       Patch size is too large Try a smaller patch

  Execution is slow       Too many model          Increase stride or
                          evaluations             patch size

  Unexpected prediction   Image content or        Inspect the top-five
                          ImageNet class mapping  predictions

  Negative heatmap values Masking increased       Explain that masking
                          target probability      can sometimes remove
                                                  distracting evidence
  -----------------------------------------------------------------------

If the heatmap looks blank, do not immediately assume that the code
failed. First inspect:

``` python
print("Minimum:", heatmap.min())
print("Maximum:", heatmap.max())
print("Mean:", heatmap.mean())
```

A low score range may simply indicate that the selected target class is
not very sensitive to the tested patches.

## 18. Limitations of occlusion

Explain these limitations before concluding the demonstration.

1.  **Computational cost:** Every patch requires another model
    prediction. Large images and small strides can make the process
    slow.
2.  **Patch-size dependency:** The heatmap changes when patch size
    changes.
3.  **Replacement-value dependency:** Grey, black, white or blurred
    patches can produce different predictions.
4.  **Feature interactions:** Two regions may work together. Hiding them
    individually may not reveal their combined importance.
5.  **Not causal proof:** A score drop indicates model sensitivity to
    the masking intervention, not proof of real-world causation.
6.  **Out-of-distribution inputs:** A grey square may create an image
    pattern that the model did not encounter during training.

**Important distinction:** Occlusion is model-agnostic in the sense that
it needs only the model's input and output predictions. It does not
require gradients or access to internal layers.

## 19. Final discussion and knowledge check

Ask students to answer these questions without looking at the code.

1.  What is occlusion sensitivity?
2.  Why do we record the original class probability?
3.  What does a large positive score drop indicate?
4.  Why do we keep the target class fixed?
5.  How does patch size affect the explanation?
6.  Why can grey-patch replacement influence the result?
7.  Does an occlusion heatmap prove that the model is correct?
8.  Why is occlusion called a model-agnostic technique?

### Suggested answers

  -----------------------------------------------------------------------
  Question                            Expected answer
  ----------------------------------- -----------------------------------
  What is occlusion?                  Hiding input regions and measuring
                                      prediction changes

  Why record the original score?      It is the reference for calculating
                                      score drops

  Large positive drop?                Hiding that region reduced
                                      target-class confidence

  Why fix the target?                 To compare the same class across
                                      all occlusion tests

  Patch-size effect?                  It changes spatial resolution and
                                      the amount of information hidden

  Why does fill value matter?         Different replacement values create
                                      different modified inputs

  Does it prove correctness?          No, it measures sensitivity, not
                                      correctness

  Why model-agnostic?                 It only needs inputs and model
                                      predictions, not model internals
  -----------------------------------------------------------------------

## 20. Lab conclusion

In this lab, we used a pre-trained ResNet-18 model to classify a dog
photograph. We then systematically covered image regions with grey
patches and measured how the predicted class probability changed.

The resulting heatmap helped us investigate which image regions
influenced the model's prediction.

### Key takeaway

**A prediction tells us what the model decided. Occlusion helps us
investigate which parts of the input influenced that decision.**

### Link to the next lab -- Grad-CAM

In this lab, we treated the neural network as a black box and observed
how its output changed when we modified the input.

In Lab 2, we will go one step further. We will use gradients and
internal CNN feature maps to generate a **Grad-CAM heatmap**.

Ask students:

> "If occlusion requires us to hide hundreds of image patches one by
> one, can we use the neural network's internal information to generate
> an explanation more directly?"

That question introduces Grad-CAM.

## References

-   [Google Colab](https://colab.research.google.com/)
-   [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
-   [Torchvision ResNet-18
    documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
-   [PyTorch Hub sample dog
    image](https://github.com/pytorch/hub/blob/master/images/dog.jpg)
-   [Matplotlib documentation](https://matplotlib.org/stable/index.html)
