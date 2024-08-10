# Unet
- python Unet&Mask2Former/train.py unet_config.py

# Mask2Former
- python Unet&Mask2Former/train.py mask2former_config.py
  
# nnUnet
- cd nnunet
- pip install -e .
- nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD 
- nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD --val --npz
- nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD 
