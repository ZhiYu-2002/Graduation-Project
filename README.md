# Graduation-Project
Laparoscopic Surgical Instruments Binary Segmentation (2023 fall - 2024 spring)

• Designed a novel deep learning network based on U-Net architecture with Transformer, and tested on the Endovis2017 dataset for surgical instruments binary segmentation.

• Implemented ablation experiments to verify the efficiency of each part of the network including attention mechanism and residual blocks.

• Segmented the sequenced images extracted from the video, approximately achieved the surgical instrument tracking and performed faster compared to others.

## Environment

• All network structures in this article are implemented in Python programming language and implemented using PyTorch library.

• Network training and verification are performed on Dell R740 server. The server is equipped with 2 Intel Xeon ® Gold 5218 CPU @ 2.10 GHz processors, 128 GB memory and a NVIDIA TESLA A10 computing card with 24 G video memory.

## Final Visualization Result

https://github.com/user-attachments/assets/f11b9d33-4d87-4e0c-9385-3872a78412ea

## Proposed Network Architecture

![Network Architecture](https://github.com/user-attachments/assets/4ce29fbf-6b70-47e3-a8e2-4f304c79de82)

## Notes

• Code is modified based on m2caiSeg work: https://github.com/salmanmaq/segmentationNetworks.

• If utilizing the source code to display samples directly, the original labels with 3 channels (L mode) have to be converted to 1 channel. Please be careful. Sorry for this coarse work.

• Different versions distribute in these zip files with causal names (sorry about that), and to the best of my knowledge, the 4th version performed best. Please check again before use since the network performances probably vary according to deployment. 

## Conclusion

Versatile code is available.

• Visualised the image, groundtruth, and segmentation in training and validation.

• Made image and groundtruth appear in pairs as a supervised learning.

• Used IoU, precision, recall, and F1 score as metrics.

• Saved the trained model, load it for validation and calculate FPS to test the real-time performance.

• Compared the proposed model with other existing deep learning models.

## Contact

• Any question or original outcome requirement please forward yuzhi0327 at gmail dot com, thank you.
