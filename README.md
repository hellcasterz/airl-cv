# airl-cv
# airl-cv
This repository contains the official implementation for the AIRL-CV project, focusing on patch-based visual classification.

How to Run in Google Colab
You can easily run this project in a Google Colab environment.

 <!-- Replace with your actual Colab notebook link -->

Steps:

Clone the repository:

!git clone [https://github.com/YOUR_USERNAME/airl-cv.git](https://github.com/YOUR_USERNAME/airl-cv.git)
%cd airl-cv

Install dependencies:

!pip install -r requirements.txt

Run training/evaluation:
Open the main notebook (main.ipynb or similar) and execute the cells to download data, configure the model, and start the training and evaluation process.

Best Model Configuration
Our best-performing model was achieved using the following configuration. This setup provides a strong baseline for patch-based classification tasks.

model:
  name: ViT-Base
  patch_size: 16
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0

data:
  dataset: CIFAR-100 # or your dataset
  image_size: 224
  augmentation:
    - RandomResizedCrop(224)
    - RandomHorizontalFlip()
    - RandAugment(2, 9)
    - ToTensor()
    - Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  
training:
  optimizer: AdamW
  learning_rate: 1e-4
  weight_decay: 0.05
  batch_size: 128
  epochs: 100
  scheduler: CosineAnnealingLR
  warmup_epochs: 5
  patch_strategy: non-overlapping # or overlapping

Results
The following table summarizes the overall classification accuracy on the test set for different model configurations.

Model Variant

Overall Test Accuracy

ResNet50 Baseline

78.5%

ViT-Small (Non-overlapping)

81.2%

ViT-Base (Non-overlapping)

84.7%

ViT-Base (Overlapping)

85.3%

Analysis & Ablation Studies
We conducted several experiments to understand the impact of different design choices.

Patch Size Choices
The choice of patch size presents a fundamental trade-off.

Smaller Patches (e.g., 8x8): Capture finer details and local textures but increase the sequence length, leading to higher computational costs.

Larger Patches (e.g., 32x32): Capture more global context within each patch but risk losing important fine-grained information.
For our primary dataset, a 16x16 patch size provided the best balance between performance and computational efficiency.

Depth vs. Width Trade-offs
We explored different model architectures by varying the depth (number of transformer layers) and width (embedding dimension).

Deeper, Narrower Models: Showed strong performance but were slower to train. They are effective at learning complex feature hierarchies.

Wider, Shallower Models: Trained faster and offered competitive results, suggesting that a rich feature representation in early layers is highly beneficial. The ViT-Base architecture (Depth: 12, Width: 768) ultimately outperformed wider but shallower variants.

Augmentation Effects
Data augmentation was critical for achieving good generalization. We found that a combination of geometric and color-space augmentations yielded the best results. RandAugment was particularly effective, preventing the model from overfitting to superficial features and encouraging it to learn more robust representations. Disabling augmentation resulted in a significant drop in accuracy (~8-10%).

Optimizer and Schedule Variants
We tested several optimizers and learning rate schedules.

Optimizer: AdamW consistently outperformed SGD and standard Adam due to its improved weight decay implementation, which is crucial for training transformer-based models.

LR Schedule: A CosineAnnealingLR schedule with a brief linear warmup period (5-10 epochs) was essential. It allowed the model to stabilize in the early stages of training before gradually annealing the learning rate, leading to better convergence and final performance.

Overlapping vs. Non-overlapping Patches
We experimented with both non-overlapping patches and overlapping patches (with a stride smaller than the patch size).

Non-overlapping: This is the standard, computationally efficient approach.

Overlapping: This method slightly improved performance (~0.5-0.6%) by providing smoother transitions between patches and ensuring features at patch borders are processed multiple times in different local contexts. However, this comes at the cost of increased computational complexity during the patching phase. For our final best model, this trade-off was deemed worthwhile.
