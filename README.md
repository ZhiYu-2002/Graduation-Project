# Graduation-Project
Laparoscopic Surgical Instruments Binary Segmentation (2023 fall - 2024 spring)

• Designed a novel deep learning network based on U-Net architecture with Transformer, and tested on the Endovis2017 dataset for surgical instruments binary segmentation

• Implemented ablation experiments to verify the efficiency of each part of the network including attention mechanism and residual blocks

• Segmented the sequenced images extracted from the video, approximately achieved the surgical instrument tracking and performed faster compared to others

## Final Visualization Result

https://github.com/user-attachments/assets/f11b9d33-4d87-4e0c-9385-3872a78412ea

## Proposed Network Architecture

![Network Architecture](https://github.com/user-attachments/assets/4ce29fbf-6b70-47e3-a8e2-4f304c79de82)

## Notes

Versatile code is available.

• Visualise the image, groundtruth, and segmentation in training and validation.

• Make image and groundtruth appear in pairs as a supervised learning.

• Used IoU, precision, recall, F1 score as metrics.

• Save the trained model, load it for validation and calculate FPS to test the real time performance.

• Compared the proposed model with other existing deep learning model.
