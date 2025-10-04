# AIRL-CV
This repository contains the implementation of ViT on CIFAR 10 dataset. I used WandB to track accuracy and loss during training process.

Final Accuracy: 
<img width="1177" height="346" alt="image" src="https://github.com/user-attachments/assets/0c0195e8-1cad-4cee-be5e-88016c830dd5" />

## Vision Transformer on CIFAR 10 dataset

<!-- Replace with your actual Colab notebook link -->
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1R_7xdv9hryOd1eeMQZFFb2EVEbaYM4si#scrollTo=tKJmfP9uIunk)

### Steps:

1. **Clone the repository**
   ```bash
   !git clone https://github.com/YOUR_USERNAME/airl-cv.git
   %cd airl-cv

   
Run training/evaluation:
Open the main notebook (q1.ipynb) and execute the cells to download data, configure the model, and start the training and evaluation process.

Best Model Configuration
Our best-performing model was achieved using the following configuration. This setup provides a strong baseline for patch-based classification tasks.

2. **Best Performing Model**
   ```bash
   name: ViT
   patch_size: 4
   embed_dim: 256
   num_heads: 8
   block_size: 8

3. **Data**
   ```bash
   dataset: CIFAR-10
   image_size: 32
   augmentation:
      - RandomCrop(IMAGE_SIZE, padding=4),
      - RandomHorizontalFlip(),
      - ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
      - RandomAffine(degrees=25, translate=(0.1, 0.1)),
      - ToTensor(),
      - Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      - RandomErasing(p=0.5, scale=(0.05, 0.25), ratio=(0.3, 3.3), value=0)
 
  4. **Training:**
     ```bash
     optimizer: AdamW
     learning_rate: 5e-2
     weight_decay: 0.05
     batch_size: 128
     epochs: 80
     scheduler: CosineAnnealingLR
     patch_strategy: non-overlapping
<img width="1170" height="342" alt="image" src="https://github.com/user-attachments/assets/87025157-7346-4223-b6d8-667b2adb9fd5" />


   ## Analysis

- **Patch Size Choices**
  - Larger patch sizes risk losing important fine-grained information.  
  - Since CIFAR images are low resolution, large patches were not feasible.  
  - Smaller patch sizes capture finer details but increase computation.  
  - Due to local implementation and time constraints, a **patch size of 4** was chosen as the best balance.

- **Augmentation Effects**
  - Data augmentation was **critical** for good generalization.  
  - A combination of geometric and color-space augmentations gave the best results.  
  - **ColorJitter** and **RandomAffine** were most effective.  
  - Without augmentation, accuracy peaked at only **53% after 60 epochs**.

- **Optimizer**
  - **AdamW** outperformed standard Adam due to better weight decay handling.

- **Learning Rate Schedule**
  - **CosineAnnealingLR** with 5–10 warmup epochs stabilized training and improved convergence.  
  - As epochs were in the range of 40–80, **MultiStepLR** produced less smooth curves.  
  - However, **71% accuracy** was still achieved with MultiStepLR.


 # Grounding SAM
 I have read both SAM and SAM 2 papers available and also read about CLIP, DINO, and Grounding DINO, and tried to work with them, but I was not able to run implement them in the gviven time constraints.
