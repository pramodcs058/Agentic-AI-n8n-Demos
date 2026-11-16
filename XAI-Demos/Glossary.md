# Explainability in Deep Neural Networks (XAI)
## Complete Key Terms Glossary

This glossary covers the important technical concepts, algorithms, architectures, tools, and terminology from the 40-slide presentation **Explainability in Deep Neural Networks (XAI)**. Each definition is kept concise for trainer preparation, classroom explanation, and quick revision.

## 1. Fundamentals of Explainable AI

| Key term | Explanation |
|---|---|
| **Artificial Intelligence (AI)** | Technology that enables machines to perform tasks that normally require human intelligence, such as reasoning, prediction, and decision-making. |
| **Explainable AI (XAI)** | A set of techniques that help humans understand why an AI model produced a particular prediction or decision. |
| **Black-box Model** | A model whose internal decision-making process is difficult to understand, even though its inputs and outputs are visible. Deep neural networks are common examples. |
| **Interpretable AI** | AI models designed so humans can directly understand how they make decisions, such as linear regression and small decision trees. |
| **Explainability** | The ability to provide understandable reasons for a model's predictions, often using additional explanation techniques after training. |
| **Interpretability** | How easily a person can understand the internal logic and behaviour of a model directly. |
| **Post-hoc Explainability** | Explaining a trained model after it has been developed, without changing its original architecture or training process. |
| **Global Explanation** | Explains the overall behaviour of a model, such as which features are most important across the entire dataset. |
| **Local Explanation** | Explains why a model produced a particular prediction for one specific input or individual. |
| **Model-specific Method** | An explanation technique that uses the internal architecture, layers, weights, or gradients of a particular model. |
| **Model-agnostic Method** | An explanation technique that treats the model as a black box and generally requires only inputs and outputs. |
| **Feature** | An individual input variable used by a machine learning model to make predictions, such as age, income, or temperature. |
| **Feature Attribution** | A technique that assigns importance or contribution scores to input features to show how they influence a prediction. |
| **Prediction Confidence** | A model's numerical indication of how strongly it supports a particular predicted class. It is not necessarily a measure of actual correctness. |
| **Shortcut Learning** | When a model learns an easier but potentially unreliable pattern instead of the intended meaningful characteristics of the data. |
| **Accuracy** | The percentage of predictions that a model correctly classifies out of all predictions. High accuracy alone does not guarantee trustworthy decisions. |

## 2. Adversarial Features and Data Augmentation

| Key term | Explanation |
|---|---|
| **Adversarial Example** | An input deliberately modified with a small perturbation that causes a model to produce an incorrect prediction. |
| **Adversarial Features** | Input characteristics or pixels to which the model is highly sensitive, potentially allowing small changes to alter its prediction. |
| **Perturbation** | A small modification introduced into an input, such as adding minor noise to an image. |
| **Adversarial Attack** | An attempt to manipulate a model's prediction by introducing carefully designed changes to its input. |
| **FGSM (Fast Gradient Sign Method)** | A technique that creates adversarial examples by changing input pixels in the direction indicated by the sign of the loss gradient. |
| **Gradient** | A mathematical quantity that indicates how much a model's output or loss changes with respect to an input variable. |
| **Loss Function** | A mathematical function that measures how far a model's prediction is from the expected answer. |
| **Epsilon (ε)** | A parameter that controls the size of the perturbation applied during an adversarial attack. |
| **Gradient Sign** | The positive or negative direction of a gradient, used by FGSM to determine how to modify input pixels. |
| **Robustness** | The ability of a model to maintain reliable predictions when its inputs contain small changes, noise, or disturbances. |
| **Data Augmentation** | Creating modified versions of training data using transformations such as rotation, flipping, cropping, and brightness changes. |
| **Rotation** | An image transformation that turns an image by a specified angle to expose a model to different orientations. |
| **Flipping** | Reversing an image horizontally or vertically to create additional training examples. |
| **Cropping** | Removing part of an image to focus on a particular region or generate different image views. |
| **Explanation Stability** | The consistency of an explanation when the input undergoes harmless transformations. |
| **Noise** | Random or unwanted variations added to data, sometimes used to test the sensitivity of a model. |

## 3. Occlusion and Image Explainability

| Key term | Explanation |
|---|---|
| **Occlusion** | A technique that hides or masks part of an input and measures how much the model's prediction changes. |
| **Occlusion Sensitivity** | Measures how sensitive a model's prediction is to hiding different regions of an input. |
| **Occlusion Map** | A visual representation showing which image regions caused the greatest change in the model's prediction when hidden. |
| **Masking** | Replacing a selected portion of an input with another value or representation to examine its influence on the prediction. |
| **Patch** | A small region of an image selected for occlusion analysis. |
| **Stride** | The number of pixels by which an occlusion patch moves during the scanning process. |
| **Heatmap** | A visual representation that uses different colours or intensities to indicate the relative importance of different regions. |
| **Saliency** | The degree to which an input region influences a model's prediction. |
| **Score Drop** | The decrease in a model's prediction score after a particular input region is hidden or modified. |

## 4. Convolutional Neural Networks and Layer-wise Analysis

| Key term | Explanation |
|---|---|
| **Deep Neural Network (DNN)** | A neural network containing multiple layers that learn increasingly complex patterns from input data. |
| **CNN (Convolutional Neural Network)** | A deep learning architecture commonly used for image processing and computer vision by identifying spatial patterns. |
| **Convolutional Layer** | A CNN layer that applies filters to input data to detect patterns such as edges, textures, and shapes. |
| **Filter / Kernel** | A small matrix of learnable values that moves across an image to detect specific patterns. |
| **Feature Map** | The output generated by applying a convolutional filter, representing detected patterns at different image locations. |
| **Activation** | The output produced by a neuron or neural network layer after applying an activation function. |
| **Early Layers** | Initial CNN layers that generally learn simple patterns, such as edges, lines, and colours. |
| **Intermediate Layers** | CNN layers that combine simple patterns to recognise more complex structures, such as textures and object parts. |
| **Deep Layers** | Later CNN layers that learn higher-level patterns associated with objects, faces, and other complex structures. |
| **CAM (Class Activation Mapping)** | A technique that creates a heatmap showing which image regions contribute to a particular class prediction. |
| **Global Average Pooling (GAP)** | A pooling operation that calculates the average value of each feature map, reducing its spatial dimensions to one value. |
| **Class Weights** | Learned weights that determine how strongly different feature maps contribute to a particular output class in CAM. |
| **Weighted Sum** | A mathematical operation that multiplies different values by their respective importance weights and adds the results. |
| **Grad-CAM (Gradient-weighted Class Activation Mapping)** | An explanation technique that uses gradients of a target class score to generate a heatmap highlighting important image regions. |
| **Class Score** | The numerical output produced by a model for a particular class before or after probability conversion. |
| **ReLU (Rectified Linear Unit)** | An activation function that outputs zero for negative values and keeps positive values unchanged. Grad-CAM uses ReLU to retain regions that positively support the target class. |
| **Upsampling** | Increasing the spatial resolution of a smaller feature map so that it can be displayed over the original image. |
| **Forward Pass** | The process of passing input data through a neural network to calculate its prediction. |
| **Backpropagation** | The process of calculating gradients by propagating error information backward through the network. |
| **Gradient-based Explanation** | An explanation approach that uses gradients to estimate how input features or internal activations influence an output. |
| **Hook** | A PyTorch mechanism that allows a developer to capture information, such as feature maps or gradients, during model execution. |
| **Forward Hook** | A hook that captures information during the forward pass, such as the output of a selected CNN layer. |
| **Backward Hook** | A hook that captures gradient information during backpropagation. |
| **Pre-trained Model** | A model that has already been trained on a dataset and can be reused for prediction or further training. |
| **ResNet-18** | An 18-layer Residual Network architecture commonly used for image classification and demonstrations of CNN explainability. |
| **Residual Network (ResNet)** | A CNN architecture that uses shortcut connections to help information and gradients flow through deeper networks. |

## 5. SHAP and DeepLIFT

| Key term | Explanation |
|---|---|
| **SHAP (SHapley Additive exPlanations)** | An explainability framework that assigns contribution values to individual features to explain a prediction. |
| **Shapley Values** | Values from cooperative game theory that distribute a total contribution among participants based on their average marginal contributions across possible groupings. |
| **Game Theory** | A mathematical framework for analysing interactions between multiple participants whose actions influence a shared outcome. |
| **Feature Contribution** | The amount by which an individual feature contributes to increasing or decreasing a model's prediction relative to a reference value. |
| **Base Value** | The reference prediction used as the starting point when calculating individual feature contributions in SHAP explanations. |
| **Positive SHAP Value** | Indicates that a feature pushes the model's prediction higher relative to the baseline. |
| **Negative SHAP Value** | Indicates that a feature pushes the model's prediction lower relative to the baseline. |
| **DeepSHAP** | An approximation of SHAP for deep learning models that combines DeepLIFT-style propagation rules with Shapley-value ideas. |
| **Background Dataset** | A representative set of input samples used as a reference when computing SHAP or DeepSHAP explanations. |
| **DeepLIFT (Deep Learning Important FeaTures)** | A technique that explains predictions by comparing neuron activations against reference activations and propagating contribution scores backward. |
| **Reference Input** | A baseline input used by DeepLIFT to determine how the actual input differs from a reference example. |
| **Reference Activation** | The activation generated by a neuron when the reference input is passed through the model. |
| **Contribution Score** | A numerical value indicating how much an input feature or neuron contributes to the difference between the actual and reference outputs. |
| **Activation Difference** | The difference between a neuron's activation for the actual input and its activation for the reference input. |
| **Saturation** | A situation in which a neuron or activation function changes very little in response to input changes, potentially making ordinary gradients uninformative. |
| **Flat ReLU** | A region where ReLU produces zero output for negative inputs and has zero gradient, making gradient-based attribution difficult in that region. |
| **Additivity** | A property in which individual feature contributions sum to the difference between the model's prediction and its reference or baseline prediction. |

## 6. Counterfactual Explanations, Causality, and CEM

| Key term | Explanation |
|---|---|
| **Counterfactual Explanation** | Describes what minimal change to an input could produce a different model prediction. |
| **What-if Analysis** | An approach that changes input variables and observes how the model's prediction responds. |
| **Actionable Features** | Input variables that a person can realistically change, such as income or loan amount. |
| **Non-actionable Features** | Input variables that a person cannot reasonably change, such as age or birthplace. |
| **Validity** | A measure of whether a counterfactual actually produces the intended alternative prediction. |
| **Proximity** | Measures how close the counterfactual input is to the original input. |
| **Sparsity** | Refers to changing as few input features as possible when constructing a counterfactual explanation. |
| **Causal Inference** | The process of identifying cause-and-effect relationships and estimating what happens when variables are deliberately changed. |
| **Correlation** | A statistical relationship indicating that two variables tend to change together. It does not establish that one causes the other. |
| **Causation** | A cause-and-effect relationship where changing one variable directly or indirectly influences another. |
| **Causal Graph** | A diagram representing variables as nodes and causal relationships as directed arrows between them. |
| **Confounder** | A third variable that influences two other variables, creating or affecting an observed association between them. |
| **Intervention** | Deliberately changing a variable to examine its causal effect on another variable. |
| **Proxy Feature** | A feature that indirectly represents another characteristic, potentially including a sensitive or protected attribute. |
| **CEM (Contrastive Explanation Method)** | An explanation technique that identifies which features must be present or absent to support or change a model prediction. |
| **Pertinent Positives (PP)** | The minimum relevant features that must remain present to support a particular prediction. |
| **Pertinent Negatives (PN)** | Features that must be absent, because their presence could alter the current prediction. |
| **Contrastive Explanation** | Explains why a particular outcome occurred instead of another possible outcome. |

## 7. Google's What-If Tool and Fairness

| Key term | Explanation |
|---|---|
| **What-If Tool (WIT)** | An interactive visual tool used to investigate trained machine learning models by editing inputs, exploring predictions, finding counterfactuals, and examining fairness. |
| **Data Slicing** | Dividing a dataset into meaningful groups, such as gender, region, or age categories, to compare model behaviour. |
| **Fairness Audit** | An evaluation process that checks whether an AI model treats different groups appropriately and avoids unfair discrimination. |
| **Fairness Metric** | A numerical measure used to evaluate differences in model outcomes across demographic or other groups. |
| **Bias** | A systematic tendency in data, model design, or predictions that can lead to unfair or inaccurate outcomes. |
| **Discrimination** | Unfair treatment of individuals or groups based on characteristics that should not negatively influence a decision. |

## 8. Time-Series Explainability and LSTM

| Key term | Explanation |
|---|---|
| **Time Series** | A collection of observations recorded sequentially over time, such as hourly electricity usage or daily stock prices. |
| **Temporal Feature Attribution** | Assigning importance scores to different time steps or features to identify which historical observations influenced a prediction. |
| **Timestamp** | A value identifying the specific date and time associated with an observation. |
| **LSTM (Long Short-Term Memory)** | A type of recurrent neural network designed to learn long-term dependencies in sequential data. |
| **RNN (Recurrent Neural Network)** | A neural network architecture designed to process sequential data by carrying information from previous time steps. |
| **Cell State** | The internal memory of an LSTM that carries information across multiple time steps. |
| **Forget Gate** | An LSTM component that determines which information from the previous cell state should be discarded. |
| **Input Gate** | An LSTM component that determines which new information should be stored in its memory. |
| **Output Gate** | An LSTM component that determines which information from the cell state should contribute to the current output. |
| **Gate Values** | Numerical values calculated by LSTM gates to control how information flows through the network. |
| **Sequence** | An ordered collection of observations, such as 24 hourly electricity readings. |
| **Time-step Attribution** | An importance score that indicates how much a particular observation in a sequence contributes to the final prediction. |
| **Forecasting** | Predicting future values using historical observations, such as predicting tomorrow's electricity demand. |

## 9. Integrated Gradients

| Key term | Explanation |
|---|---|
| **Integrated Gradients (IG)** | An attribution technique that measures feature contributions by accumulating gradients along a path from a baseline input to the actual input. |
| **Baseline Input** | A reference input, such as an all-zero series or a black image, against which the actual input's contributions are measured. |
| **Path** | The sequence of intermediate inputs connecting the baseline to the actual input during Integrated Gradients calculation. |
| **Gradient Integration** | Accumulating gradient values across multiple points along the path to estimate a feature's total contribution. |
| **Completeness** | A property of Integrated Gradients stating that the sum of feature attributions should equal the difference between the actual and baseline predictions. |
| **Sensitivity** | A property requiring an explanation method to recognise an input feature that meaningfully affects the model's output. |
| **Convergence Delta** | A numerical measure used by Captum to assess how closely the calculated attributions satisfy the completeness property. A value close to zero generally indicates better convergence. |
| **Captum** | An open-source PyTorch library that provides tools for interpreting and explaining machine learning and deep learning models. |
| **Attribution Score** | A numerical value assigned to an input feature to represent its contribution to a model prediction. |

## 10. Transformers and Large Language Model Explainability

| Key term | Explanation |
|---|---|
| **Transformer** | A neural network architecture that uses attention mechanisms to process relationships between elements in a sequence. |
| **Token** | A unit of text processed by a language model; it may be a complete word, part of a word, or punctuation. |
| **Tokenization** | The process of converting text into tokens that a language model can process. |
| **Embedding** | A numerical vector representation of a token that captures information about its meaning and relationships. |
| **Positional Encoding** | Information added to token representations to help a Transformer understand the order or position of tokens in a sequence. |
| **Self-Attention** | A mechanism that allows each token to consider information from other tokens in the input sequence. |
| **Query (Q)** | In attention, a vector representing what a token is looking for in other tokens. |
| **Key (K)** | A vector representing the information a token makes available for comparison with queries. |
| **Value (V)** | A vector containing the information that is combined to produce the attention output. |
| **Query-Key Similarity** | A score measuring how strongly a query relates to another token's key; it helps determine attention weights. |
| **Attention Weight** | A numerical value indicating how much attention a token assigns to another token during the attention calculation. |
| **Multi-Head Attention** | A mechanism that performs multiple attention calculations in parallel, allowing the model to capture different relationships between tokens. |
| **Feed-Forward Network** | A neural network applied to token representations after attention to further process and transform their information. |
| **Residual Connection** | A shortcut connection that adds an earlier layer's input to a later layer's output, helping information and gradients flow through deep networks. |
| **Attention Map** | A visual representation of attention weights showing how tokens relate to other tokens in a sequence. |
| **Attention Rollout** | A technique that combines attention information across multiple Transformer layers to estimate how information flows from earlier tokens. |
| **Token Attribution** | Assigning importance scores to individual tokens to understand how they influence a particular model prediction. |
| **Hidden State** | An internal vector representation generated by a neural network layer that contains information learned from the input. |
| **Probing** | Analysing the internal representations of a model to investigate what types of information its layers encode. |
| **Hallucination** | A situation where a generative AI model produces information that appears plausible but is incorrect, unsupported, or fabricated. |
| **Negation** | A word or expression that reverses or denies meaning, such as changing “good” to “not good”. |
| **Sentiment Analysis** | A natural language processing task that identifies the emotional tone or opinion expressed in text, such as positive or negative sentiment. |
| **Language Model (LM)** | A model trained to understand or predict sequences of text. |
| **Large Language Model (LLM)** | A large-scale language model trained on extensive text data to perform tasks such as answering questions, summarising, and generating content. |
| **Prompt Attribution** | Analysing which parts of a prompt influence the output generated by an AI model. |
| **DistilBERT** | A smaller, more efficient Transformer-based language model derived from BERT through a process called knowledge distillation. |
| **Knowledge Distillation** | A model compression technique in which a smaller student model learns to approximate the behaviour of a larger teacher model. |

## 11. Tools, Libraries, and Development Terminology

| Key term | Explanation |
|---|---|
| **Google Colab** | A cloud-based notebook environment where developers can write and execute Python code without setting up a complete local environment. |
| **PyTorch** | An open-source deep learning framework used to build, train, and execute neural networks. |
| **torchvision** | A PyTorch library providing datasets, image transformations, and pre-trained computer vision models. |
| **NumPy** | A Python library for numerical computing, particularly array and matrix operations. |
| **Matplotlib** | A Python visualisation library used to create plots, graphs, and heatmaps. |
| **Captum** | A PyTorch interpretability library supporting techniques such as Integrated Gradients, DeepLIFT, and layer attribution. |
| **Hugging Face Transformers** | A Python library providing pre-trained Transformer models and tools for natural language processing tasks. |
| **AutoTokenizer** | A Hugging Face component that automatically loads the tokenizer associated with a particular pre-trained model. |
| **Sequence Classification** | A task that assigns a label to an entire input sequence, such as classifying a sentence as positive or negative sentiment. |
| **Softmax** | A mathematical function that converts a vector of values into a probability distribution across classes. |
| **Logits** | Raw numerical scores produced by a classification model before applying a function such as Softmax. |
| **Model Evaluation Mode (`eval()`)** | A PyTorch model mode used during inference, changing the behaviour of layers such as Dropout and Batch Normalisation. |
| **Autograd** | PyTorch's automatic differentiation engine, which calculates gradients required for training and gradient-based explanations. |
| **SST-2** | The Stanford Sentiment Treebank binary classification dataset used for positive and negative sentiment analysis. |

## 12. Responsible AI and Industry Terminology

| Key term | Explanation |
|---|---|
| **Responsible AI** | Designing and operating AI systems in ways that promote fairness, transparency, accountability, reliability, and privacy. |
| **Transparency** | Making AI system behaviour, decisions, and limitations understandable to relevant stakeholders. |
| **Accountability** | Ensuring that people or organisations can be held responsible for the outcomes and consequences of AI systems. |
| **Privacy** | Protecting personal and confidential information used by AI systems and exposed through their explanations. |
| **Trustworthy AI** | AI that demonstrates appropriate reliability, safety, fairness, transparency, and accountability. |
| **Human Oversight** | Keeping people involved in reviewing, monitoring, or approving AI decisions, particularly in high-impact applications. |
| **Explainability Risk** | The possibility that an explanation may be misleading, incomplete, or interpreted as stronger evidence than it actually provides. |
| **Shortcut Feature** | An irrelevant or indirect input characteristic that a model learns to use instead of meaningful evidence. |
| **Model Audit** | A systematic examination of a model's behaviour, performance, fairness, reliability, and potential risks. |
| **High-stakes AI** | AI systems used in decisions that can significantly affect people's lives, such as healthcare, finance, recruitment, and justice. |

## Quick Revision: Which Technique Answers Which Question?

| Technique | Main question it answers |
|---|---|
| **FGSM** | How sensitive is the model to small adversarial changes? |
| **Data Augmentation** | Does the model behave consistently under realistic input transformations? |
| **Occlusion** | What happens when I hide a part of the input? |
| **CAM** | Which image regions support this class prediction? |
| **Grad-CAM** | Which image regions contribute most to the target class score? |
| **SHAP / DeepSHAP** | How much does each feature contribute to the prediction? |
| **DeepLIFT** | How does each feature contribute relative to a reference input? |
| **Counterfactual** | What is the smallest input change that could flip the prediction? |
| **Causal Inference** | Would changing this variable actually change the outcome in the real world? |
| **What-If Tool** | How does the model behave when I edit inputs or compare groups? |
| **CEM** | What must be present or absent to support a particular prediction? |
| **Integrated Gradients** | How much did each input feature contribute along the path from baseline to input? |
| **Transformer Token Attribution** | Which tokens influence the model's selected output? |
| **Attention Rollout** | How does attention information flow across Transformer layers? |

## Important Teaching Distinction

**Feature attribution** explains what the model used; **counterfactual explanations** describe what could change the prediction; **causal inference** investigates what actually causes an outcome. These three concepts are related, but they are not interchangeable.

---

*Prepared as a teaching and revision reference for the presentation “Explainability in Deep Neural Networks (XAI)”.*
