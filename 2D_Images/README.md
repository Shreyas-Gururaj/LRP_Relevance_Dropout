console.log(`# 2D Image Classification with RelDrop

## Directory Structure
\`\`\`
./2D_Images/
├── run_scripts/
│   └── cifar_imagenet.py               # Training script for CIFAR and ImageNet
│   └── custom_transforms.py            # Random Erasing per image
│   └── relevance_augmentations.py      # Batch level Relevance augmentation - RelDrop (pixel, block, patch)
│   └── dataset_dir.py                  # Required for ImageNet dataloader
│   └── training_config.json            # All the parameters related to dataset, model, transformations, optimizer, scheduler, etc. can be set here
│   └── utils/                          # Utility functions required for training 
├── zero_shot_eval/
│   ├── eval_imagenet_a_o.py            # Evaluation script for ImageNet-A/O
│   └── eval_imagenet_r.py              # Evaluation script for ImageNet-R
│   └── eval_config.json                # All the parameters related to loading and evaluating ImageNet_a, ImageNet_o and ImageNet_r can be controlled from here
└── README.md                           # This file

\`\`\`

## Training/Fine-tuning
### CIFAR-10/100 and ImageNet-1k
\`\`\`
cd ./2D_Images/run_scripts
python cifar_imagenet.py --config_file training_config.json
\`\`\`

### Zero-shot Evaluation
#### ImageNet-A/O
\`\`\`
cd ./2D_Images/zero_shot_eval
python eval_imagenet_a_o.py --config_file eval_config.json 
\`\`\`

#### ImageNet-R
\`\`\`
cd ./2D_Images/zero_shot_eval
python eval_imagenet_r.py --config_file eval_config.json
\`\`\`);
