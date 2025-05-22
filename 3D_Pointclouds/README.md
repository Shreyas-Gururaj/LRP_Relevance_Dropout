## Directory Structure
```
./3D_Pointclouds/
├── train_classification.py  # Training script
└── configs/
    ├── cannonizer_map_msg.yaml     # Canonizer map for the chosen composite in the paper (ToDo, currently directly added in the ./utils/provider.py directly)
    ├── name_map_msg.yaml           # Name map for the chosen composite in the paper (ToDo, currently directly added in the ./utils/provider.py directly)
    ├── training_config.yaml        # All the parameters related to dataset, model, transformations, optimizer, scheduler, etc. can be set here
└── data_utils/                     # Utility functions required for loading ModelNet40 and ShapeNet datasets
└── models/                         # Utility functions required for setting up PointNet++ models
└── utils/                          # Utility functions required for setting up PointNet++ models
    ├── provider.py                 # Batch level Relevance augmentation (RelDrop) and other transforms
```

## Training
```
cd ./3D_Pointclouds
python train_classification.py --config_file ./config/training_config.yaml
```

