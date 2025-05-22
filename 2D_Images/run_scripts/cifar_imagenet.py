from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from pathlib import Path
import json

from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.attribution import Gradient
from zennit.core import Stabilizer # Ensure zennit is updated if this fails
from zennit.composites import SpecialFirstLayerMapComposite, COMPOSITES 
from zennit.rules import Epsilon, Norm, Pass, Flat
from zennit.types import Convolution, Activation, AvgPool, Linear as AnyLinear, BatchNorm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as torchvision_transforms 
from torchvision.transforms import ToTensor, Normalize
# Ensure 'custom_transforms.py' (Random Erasing in Random Erasing Data Augmentation by Zhong et al) is in the same directory or adjust import
import custom_transforms 
import relevance_augmentations # Contains all the Relevance based augmentation strategies (pixel, block, patch)

import torchvision.datasets as datasets
# Ensure 'dataset_dir.py' is in the same directory or adjust import
from dataset_dir import ImageFolderWithPaths 
from timm import create_model
# from prettytable import PrettyTable # For count_parameters, if re-enabled
import numpy as np
import ssl
from huggingface_hub import hf_hub_download

# Ensure './utils' is in the same directory or adjust import
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p 
from accelerate import Accelerator
# import detectors

ssl._create_default_https_context = ssl._create_unverified_context

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='PyTorch CIFAR/ImageNet Training with Relevance Dropout')
# Config File Argument
parser.add_argument('--config_file', type=str, default=None, help='Path to JSON configuration file')

# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], type=str)
parser.add_argument('-j', '--workers', default=4, type=int, help='data loading workers')
parser.add_argument('--imagenet_training_dataset_path', default='./data', type=str)
parser.add_argument('--imagenet_validation_dataset_path', default='./data', type=str)

# Optimization options
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=64, type=int, help='train batchsize')
parser.add_argument('--test_batch', default=100, type=int, help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float, help='Dropout ratio for model layers (TIMM default, used for non-ResNet generic loading)')

# Optimizer parameters
parser.add_argument('--optimizer_method', default='SGD', choices=['SGD', 'AdamW'], type=str)
parser.add_argument('--lr_gamma', type=float, default=0.1, help='LR multiplier for schedulers')
parser.add_argument('--lr_step_size', type=int, default=10, help='Step size for StepLR scheduler')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument("--lr_min", default=1e-6, type=float, help="minimum lr for CosineAnnealingLR")
parser.add_argument('--fixed_lr', action='store_true', help='Use a fixed LR with no scheduler')
parser.add_argument("--lr_scheduler", default="steplr", choices=["steplr", "cosineannealinglr", "exponentiallr"], type=str)

# Checkpoints
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--local_dir', type=str, default='./reldrop_checkpoints', help='Base directory on job node for saving checkpoints')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', help='model architecture')
parser.add_argument('--pretrained', action='store_true', help='Use TIMM pretrained weights / original script pretrained logic')
parser.add_argument('--freeze', action='store_true', help='Freeze layers of pretrained model (Note: original ResNet logic freezes ALL if true)')

# Miscs
parser.add_argument('--manualSeed', type=int, default=20, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

# Original Random Erasing
parser.add_argument('--p_random_erasing', default=0, type=float, help='Original Random Erasing probability')
parser.add_argument('--sl_random_erasing', default=0.02, type=float, help='min erasing area')
parser.add_argument('--sh_random_erasing', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1_random_erasing', default=0.3, type=float, help='aspect of erasing area')

# Relevance Erasing - General
parser.add_argument("--composite_name", default="epsilon_zplus", choices=["epsilon_zplus", "epsilon_gamma_box"], type=str)
parser.add_argument('--eps_comp', default=0.001, type=float, help='Epsilon for LRP composite')

# Relevance Erasing Pixel
parser.add_argument('--relevance_dropout_pixel', action='store_true', help='Enable Relevance Pixel Dropout')
parser.add_argument('--prob_pixel', default=0.5, type=float, help='Probability for Relevance Pixel Dropout')
parser.add_argument('--alpha_pixel', default=0.3, type=float, help='Alpha for Relevance Pixel Dropout')
parser.add_argument('--beta_pixel', default=0.5, type=float, help='Beta for Relevance Pixel Dropout')

# Relevance Erasing Block
parser.add_argument('--relevance_dropout_block', action='store_true', help='Enable Relevance Block Dropout')
parser.add_argument('--prob_block', default=0.5, type=float, help='Probability for Relevance Block Dropout')
parser.add_argument('--k_centers_block', default=10, type=int, help='Num top relevant pixels as candidate centers')

# Relevance Erasing Patch
parser.add_argument('--relevance_dropout_patch', action='store_true', help='Enable Relevance Patch Dropout')
parser.add_argument('--prob_patch', default=0.5, type=float, help='Probability for Relevance Patch Dropout')
parser.add_argument('--k_patches_patch', default=20, type=int, help='Num top relevant patches to erase (also k for LRP topk)')

# Base directory for logs/results
parser.add_argument('--results_dir', type=str, default='./experiments_lrp_reldrop', help='Base directory for experiment logs and results')

args = parser.parse_args()

if args.config_file:
    print(f"Loading configuration from: {args.config_file}")
    with open(args.config_file, 'r') as f:
        config_params = json.load(f)
    arg_dict = vars(args)
    for key, value in config_params.items():
        if hasattr(args, key):
             arg_dict[key] = value 
    args = parser.parse_args(namespace=argparse.Namespace(**arg_dict))

state = {k: v for k, v in args._get_kwargs()}
use_cuda = torch.cuda.is_available()

if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if use_cuda: torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0

# Helper for LRP composite creation
def _create_lrp_composite(composite_name_arg, eps_comp_arg, dataset_name_arg, device, input_batch_shape):
    canonizers = [ResNetCanonizer()] 
    if args.arch.lower().startswith('vgg'): 
        canonizers = [VGGCanonizer()]

    if composite_name_arg == "epsilon_zplus":
        return SpecialFirstLayerMapComposite(
            layer_map=[
                (Activation, Pass()), (AvgPool, Norm()),
                (Convolution, Epsilon(epsilon=eps_comp_arg)),
                (AnyLinear, Epsilon(epsilon=eps_comp_arg)), 
                (BatchNorm, Pass()),
            ],
            first_map=[(AnyLinear, Flat())],
            canonizers=canonizers,
        )
    elif composite_name_arg == 'epsilon_gamma_box':
        if dataset_name_arg == 'imagenet':
            mean_norm, std_norm = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif dataset_name_arg == 'cifar10':
            mean_norm, std_norm = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        else: # CIFAR100
            mean_norm, std_norm = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        
        norm_fn = relevance_augmentations.BatchNormalize(mean_norm, std_norm, device=device)
        if len(input_batch_shape) != 4:
             raise ValueError(f"Expected input_batch_shape with 4 dims (B,C,H,W), got {input_batch_shape}")

        composite_kwargs = {
            'low': norm_fn(torch.zeros(*input_batch_shape, device=device)),
            'high': norm_fn(torch.ones(*input_batch_shape, device=device)),
            'canonizers': canonizers
        }
        if composite_name_arg not in COMPOSITES:
            print(f"Warning: Composite '{composite_name_arg}' not directly in zennit.COMPOSITES.")
        return COMPOSITES[composite_name_arg](**composite_kwargs)
    else:
        raise ValueError(f"Unknown composite name: {composite_name_arg}")
    
# Mapping of ResNet architectures to Hugging Face model IDs for downloadinf the Cifar10/Cifar100 pretrained models
MODEL_MAP = {
    'resnet18': {
        'cifar10': "edadaltocg/resnet18_cifar10",
        'cifar100': "edadaltocg/resnet18_cifar100"
    },
    'resnet34': {
        'cifar10': "edadaltocg/resnet34_cifar10",
        'cifar100': "edadaltocg/resnet34_cifar100"
    },
    'resnet50': {
        'cifar10': "edadaltocg/resnet50_cifar10",
        'cifar100': "edadaltocg/resnet50_cifar100"
    },
    'resnet101': {
        'cifar10': "edadaltocg/resnet101_cifar10",
        'cifar100': "edadaltocg/resnet101_cifar100"
    }
}

def load_pretrained_resnet_cifar(arch, dataset, pretrained):
    """Loads a pretrained ResNet model from Hugging Face Hub or locally."""
    if arch not in MODEL_MAP:
        raise ValueError(f"Unsupported architecture: {arch}")
    if dataset not in MODEL_MAP[arch]:
        raise ValueError(f"Unsupported dataset: {dataset}")

    model_id = MODEL_MAP[arch][dataset]
    filename = "pytorch_model.bin"
    cache_dir = f"../../models/{arch}_{dataset}"

    # Check if model exists locally
    model_path = os.path.join(cache_dir, filename)
    if not os.path.exists(model_path):
        print(f"Downloading model {model_id} for the first time...")
        model_path = hf_hub_download(repo_id=model_id, filename=filename, cache_dir=cache_dir)
        print(f"Downloaded to: {model_path}")
    else:
        print(f"Loading model from cache: {model_path}")

    # Manually create a CIFAR-specific ResNet model
    print(f"Creating CIFAR-specific ResNet model: {arch}")
    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = create_model(
        arch,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=3,
        global_pool='avg'
    )

    # Adjust the first convolutional layer to match CIFAR input size
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # No pooling for CIFAR

    # Load the state dictionary
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Handle cases where the weights are stored inside another key
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        # Load the weights into the modified model
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")

    # Modify the final fully connected layer for CIFAR
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.weight = nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    model.fc.bias = nn.init.zeros_(model.fc.bias)

    return model

def main():
    global best_acc, args, state
    start_epoch = args.start_epoch

    # --- Path Setup ---
    arch_part = f"{args.arch}_{'pretrained' if args.pretrained else 'scratch'}"
    method_name = "Baseline_RandomErasing"
    if args.relevance_dropout_pixel and args.relevance_dropout_block: 
        method_name = f"RelDropPixelBlock_pP{args.prob_pixel}_a{args.alpha_pixel}_b{args.beta_pixel}_pB{args.prob_block}_kC{args.k_centers_block}"
    elif args.relevance_dropout_pixel:
        method_name = f"RelDropPixel_p{args.prob_pixel}_a{args.alpha_pixel}_b{args.beta_pixel}"
    elif args.relevance_dropout_block:
        method_name = f"RelDropBlock_p{args.prob_block}_kC{args.k_centers_block}"
    elif args.relevance_dropout_patch:
        method_name = f"RelDropPatch_p{args.prob_patch}_kP{args.k_patches_patch}"
    elif args.p_random_erasing == 0 and not (args.relevance_dropout_pixel or args.relevance_dropout_block or args.relevance_dropout_patch) : 
        method_name = "Baseline_NoAug"
    lrp_part = f"LRP_{args.composite_name}_eps{args.eps_comp}" if \
               (args.relevance_dropout_pixel or args.relevance_dropout_block or args.relevance_dropout_patch) else "NoLRP"
    opt_lr_part = f"opt{args.optimizer_method}_lr{args.lr}_sched{args.lr_scheduler}_wd{args.weight_decay}"
    exp_name_parts = [args.dataset, arch_part, method_name, lrp_part, opt_lr_part, f"seed{args.manualSeed}"]
    exp_folder_name = "_".join(exp_name_parts)
    results_base_dir = Path(args.results_dir)
    exp_results_dir = results_base_dir.joinpath(exp_folder_name)
    mkdir_p(exp_results_dir)
    checkpoint_base_dir = Path(args.local_dir)
    exp_checkpoint_dir = checkpoint_base_dir.joinpath(exp_folder_name)
    mkdir_p(exp_checkpoint_dir)
    args.checkpoint_path_for_saving = str(exp_checkpoint_dir)
    with open(exp_results_dir / 'commandline_args.json', 'w+', encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    # --- End Path Setup ---

    # --- Data Loading and Transforms ---
    print(f'==> Preparing dataset {args.dataset}')
    num_classes = 0
    img_size_h, img_size_w = 0, 0 
    MEAN, STD = [], []

    base_transform_list = []
    if args.dataset == 'cifar10':
        MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        num_classes = 10; img_size_h, img_size_w = 32, 32
        base_transform_list.extend([torchvision_transforms.RandomCrop(32, padding=4), torchvision_transforms.RandomHorizontalFlip()])
    elif args.dataset == 'cifar100':
        MEAN, STD = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        num_classes = 100; img_size_h, img_size_w = 32, 32
        base_transform_list.extend([torchvision_transforms.RandomCrop(32, padding=4), torchvision_transforms.RandomHorizontalFlip()])
    elif args.dataset == 'imagenet':
        MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        num_classes = 1000; img_size_h, img_size_w = 224, 224
        base_transform_list.extend([torchvision_transforms.RandomResizedCrop(224), torchvision_transforms.RandomHorizontalFlip()])
    
    base_transform_list.extend([ToTensor(), Normalize(MEAN, STD)])
    transform_train_loader_list = list(base_transform_list) 
    any_relevance_dropout_active = args.relevance_dropout_pixel or args.relevance_dropout_block or args.relevance_dropout_patch
    if not any_relevance_dropout_active and args.p_random_erasing > 0:
        transform_train_loader_list.append(custom_transforms.RandomErasing(mean=MEAN, probability=args.p_random_erasing, sl=args.sl_random_erasing, sh=args.sh_random_erasing, r1=args.r1_random_erasing))
    transform_train_loader = torchvision_transforms.Compose(transform_train_loader_list)
    if args.dataset == 'imagenet':
        transform_test_loader = torchvision_transforms.Compose([torchvision_transforms.Resize(256), torchvision_transforms.CenterCrop(224), ToTensor(), Normalize(MEAN, STD)])
    else: 
        transform_test_loader = torchvision_transforms.Compose([ToTensor(), Normalize(MEAN, STD)])

    if args.dataset == 'imagenet':
        trainset = ImageFolderWithPaths(root=args.imagenet_training_dataset_path, transform=transform_train_loader)
        testset = ImageFolderWithPaths(root=args.imagenet_validation_dataset_path, transform=transform_test_loader)
    elif args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train_loader)
        testset = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test_loader)
    else: 
        trainset = datasets.CIFAR100(root='../../data', train=True, download=True, transform=transform_train_loader)
        testset = datasets.CIFAR100(root='../../data', train=False, download=True, transform=transform_test_loader)

    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True if args.dataset=='imagenet' else False)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    # --- End Data Loading ---

    # --- Model Setup ---
    print(f"==> Creating model '{args.arch}'")
    # model = None 

    if args.arch.startswith('resnet'):
        print(f"INFO: Attempting to load ResNet model: {args.arch} (Pretrained: {args.pretrained}) for dataset: {args.dataset}")
        resnet_arch_list = ('resnet18', 'resnet34', 'resnet50', 'resnet101') 
        
        print(f"INFO: args.pretrained is True. Attempting to load pretrained weights for {args.arch}.")
        if str(args.arch) in resnet_arch_list:
            # Specific handling for ResNets from the original script
            common_resnet_kwargs = {'num_classes': num_classes, 'in_chans': 3, 'global_pool': 'avg'}

            if args.dataset == 'imagenet':
                print(f"VERIFY: Loading PRETRAINED ImageNet model: {args.arch} using TIMM with standard ResNet stem.")
                model = create_model(str(args.arch), pretrained=args.pretrained, **common_resnet_kwargs)
                print(model)
                print(f"VERIFY: TIMM model {args.arch} loaded. Re-initializing FC layer for {num_classes} classes as per original script.")
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)    
                nn.init.zeros_(model.fc.bias)
            
            elif args.dataset == 'cifar10' or 'cifar100':
                # Try loading specific edadaltocg/resnet<version>_cifar10 from Hugging Face Hub with CIFAR stem config
                hub_model_name = f"hf_hub:edadaltocg/{str(args.arch)}_{str(args.dataset)}"
                print(f"VERIFY: Attempting to load PRETRAINED CIFAR10 model from Hugging Face Hub: {hub_model_name} with CIFAR stem configuration.")
                
                try:
                    model = load_pretrained_resnet_cifar(args.arch, args.dataset, args.pretrained)
                    print(f"VERIFY: Successfully loaded and configured Hub model {hub_model_name} with CIFAR stem.")
                    print(f"VERIFY: Re-initializing FC layer for {hub_model_name} for {num_classes} classes as per original script.")
                except Exception as e_hub:
                    print(f"ERROR: Failed to load Hub model {hub_model_name}. Error: {e_hub}")
                    # Fallback 1: Try timm's internal name like "resnet18_cifar10" (might not exist)
                    print(f"VERIFY: Falling back - Attempt 2: Loading generic {args.arch} PRETRAINED on ImageNet and adapt for CIFAR10.")
                    model = create_model(str(args.arch), pretrained=args.pretrained, **common_resnet_kwargs) 
                    model.fc = nn.Linear(model.fc.in_features, num_classes) 
                    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01); nn.init.zeros_(model.fc.bias)

        else: 
                print(f"INFO: Pretrained ResNet '{args.arch}' not in specific list {resnet_arch_list}. Loading generic {args.arch} PRETRAINED on ImageNet using TIMM.")
                model = create_model(args.arch, pretrained=args.pretrained, num_classes=num_classes, drop_rate=args.drop) 
                if hasattr(model, 'fc') and model.fc.out_features != num_classes:
                    print(f"Adapting FC layer of generic pretrained {args.arch} for {num_classes} classes.")
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01); nn.init.zeros_(model.fc.bias)
                elif hasattr(model, 'classifier') and model.classifier.out_features != num_classes: 
                    print(f"Adapting classifier layer of generic pretrained {args.arch} for {num_classes} classes.")
                    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.01); nn.init.zeros_(model.classifier.bias)

        if args.freeze and model: 
            print(f"INFO: args.freeze is True. Freezing ALL parameters for PRETRAINED ResNet model {args.arch} as per original script logic.")
            for param_val in model.parameters(): 
                param_val.requires_grad = False
            print("VERIFY: All parameters of the ResNet model should now be frozen.")

    else: # Not a ResNet architecture, use general TIMM loading
        print(f"INFO: Loading generic (non-ResNet) TIMM model: {args.arch} (Pretrained: {args.pretrained})")
        model_kwargs_base = {'num_classes': num_classes, 'drop_rate': args.drop}
        model = create_model(args.arch, pretrained=args.pretrained, **model_kwargs_base)
        if args.pretrained:
            print(f"VERIFY: Loaded generic TIMM model {args.arch} with pretrained={args.pretrained}.")
        else:
            print(f"VERIFY: Loaded generic TIMM model {args.arch} from scratch (pretrained={args.pretrained}).")

        final_layer_adapted = False
        if hasattr(model, 'get_classifier'): 
            classifier = model.get_classifier()
            if classifier is not None and classifier.out_features != num_classes:
                if isinstance(classifier, nn.Linear):
                    print(f"Adapting classifier (via get_classifier) of {args.arch} for {num_classes} classes.")
                    new_classifier = nn.Linear(classifier.in_features, num_classes)
                    nn.init.normal_(new_classifier.weight, mean=0.0, std=0.01); nn.init.zeros_(new_classifier.bias)
                    model.reset_classifier(num_classes=0) 
                    assigned_new_head = False
                    for head_name in ['fc', 'classifier', 'head']:
                        if hasattr(model, head_name) or (head_name == 'fc' and not hasattr(model, 'fc')): 
                            setattr(model, head_name, new_classifier)
                            assigned_new_head = True
                            print(f"New head assigned to model.{head_name}")
                            break
                    if not assigned_new_head:
                         print(f"Warning: Could not set new classifier for {args.arch} after reset_classifier via common names.")
                    final_layer_adapted = True
                else:
                    print(f"Warning: Classifier for {args.arch} (from get_classifier) is not nn.Linear, cannot auto-adapt.")
        
        if not final_layer_adapted: 
            if hasattr(model, 'fc') and model.fc.out_features != num_classes :
                print(f"Adapting FC layer of {args.arch} for {num_classes} classes.")
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                nn.init.normal_(model.fc.weight, mean=0.0, std=0.01); nn.init.zeros_(model.fc.bias)
            elif hasattr(model, 'classifier') and model.classifier.out_features != num_classes:
                print(f"Adapting classifier layer of {args.arch} for {num_classes} classes.")
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
                nn.init.normal_(model.classifier.weight, mean=0.0, std=0.01); nn.init.zeros_(model.classifier.bias)

        if args.freeze and model: 
            print(f"INFO: args.freeze is True. Freezing backbone for generic model {args.arch}, attempting to keep classifier trainable.")
            classifier_param_names = set()
            for head_attr_name in ['fc', 'classifier', 'head']: 
                if hasattr(model, head_attr_name):
                    module = getattr(model, head_attr_name)
                    if isinstance(module, nn.Module):
                        for p_name, _ in module.named_parameters():
                            classifier_param_names.add(f"{head_attr_name}.{p_name}")
            
            if not classifier_param_names and hasattr(model, 'get_classifier') and model.get_classifier() is not None:
                 print("INFO: Using common head names (fc, classifier, head) to identify classifier parameters for freezing.")

            all_params_frozen_check = True
            for param_name, param_val in model.named_parameters():
                if param_name in classifier_param_names:
                    param_val.requires_grad = True
                    all_params_frozen_check = False
                    print(f"  VERIFY: Keeping classifier parameter trainable: {param_name}")
                else:
                    param_val.requires_grad = False
            if not classifier_param_names:
                 print("WARNING: Could not identify specific classifier parameters for non-ResNet. All parameters might be frozen if no head like 'fc', 'classifier', or 'head' is found and has parameters.")
            if all_params_frozen_check and classifier_param_names : 
                 print("WARNING: Identified classifier parameters, but all parameters appear to be frozen. Check model structure.")


    if model is None:
        raise RuntimeError(f"Model could not be created for arch: {args.arch} and dataset: {args.dataset} with pretrained={args.pretrained}")
    
    print(f'    Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'    Trainable params: {trainable_params/1e6:.2f}M')
    if args.freeze and trainable_params == 0:
        print("WARNING: args.freeze is True and all model parameters are frozen. The model will not train.")
    # --- End Model Setup ---
    
    # --- Criterion, Optimizer, Scheduler ---
    criterion = nn.CrossEntropyLoss()
    
    params_list_for_check = list(filter(lambda p: p.requires_grad, model.parameters()))
    if not params_list_for_check:
        print("WARNING: No parameters to optimize. Check model freezing logic if this is unintended.")
        optimizer = optim.SGD([torch.nn.Parameter(torch.empty(0))], lr=args.lr) 
    else:
        if args.optimizer_method == 'SGD':
            optimizer = optim.SGD(params_list_for_check, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer_method == 'AdamW':
            optimizer = optim.AdamW(params_list_for_check, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer_method}")

    scheduler = None
    if not args.fixed_lr and params_list_for_check: 
        if args.lr_scheduler == "steplr": scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr": scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
        elif args.lr_scheduler == "exponentiallr": scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    # --- End Optimizer Setup ---

    accelerator = Accelerator()
    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, testloader, scheduler
    )
    
    # --- Batch-Level Relevance Transforms ---
    batch_level_relevance_transforms = []
    _composite_creator = lambda dev, actual_batch_shape: _create_lrp_composite(args.composite_name, args.eps_comp, args.dataset, dev, actual_batch_shape)
    if args.relevance_dropout_block: batch_level_relevance_transforms.append(relevance_augmentations.BatchRelevantErasing_Block(model=model, composite_creator_fn=_composite_creator, num_classes=num_classes, mean=MEAN, probability=args.prob_block, sl=args.sl_random_erasing, sh=args.sh_random_erasing, r1=args.r1_random_erasing, k_centers_to_consider=args.k_centers_block))
    if args.relevance_dropout_patch: batch_level_relevance_transforms.append(relevance_augmentations.BatchRelevantErasing_Patch(model=model, composite_creator_fn=_composite_creator, num_classes=num_classes, mean=MEAN, probability=args.prob_patch, sl=args.sl_random_erasing, sh=args.sh_random_erasing, r1=args.r1_random_erasing, k_patches_to_erase=args.k_patches_patch))
    if args.relevance_dropout_pixel: batch_level_relevance_transforms.append(relevance_augmentations.BatchRelevantPixelDropout(model=model, composite_creator_fn=_composite_creator, num_classes=num_classes, mean=MEAN, probability=args.prob_pixel, alpha_dropout_rel=args.alpha_pixel, beta_dropout_rel=args.beta_pixel))
    # --- End Batch-Level Transforms ---

    # --- Resume, Evaluate, Training Loop ---
    title = f'{args.dataset}-{args.arch}-{method_name}' 
    logger_path = exp_results_dir / 'log.txt'
    if args.resume:
        print(f'==> Resuming from checkpoint {args.resume}')
        if not os.path.isfile(args.resume): print(f'Error: no checkpoint directory found at {args.resume}'); return
        checkpoint = torch.load(args.resume, map_location='cpu')
        accelerator.unwrap_model(model).load_state_dict(checkpoint['state_dict'])
        
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None and optimizer.param_groups and optimizer.param_groups[0]['params']:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None and optimizer.param_groups and optimizer.param_groups[0]['params']: 
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']; best_acc = checkpoint['best_acc']
        logger = Logger(logger_path, title=title, resume=True)
    else:
        logger = Logger(logger_path, title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Test Loss', 'Train Acc.', 'Test Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, accelerator)
        accelerator.print(f' Test Loss:  {test_loss:.8f}, Test Acc:  {test_acc:.2f}')
        return

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups and optimizer.param_groups[0]['params'] else 0.0
        accelerator.print(f'\nEpoch: [{epoch + 1} | {args.epochs}] LR: {current_lr:.7f}')
        
        if optimizer.param_groups and optimizer.param_groups[0]['params']:
            train_loss, train_acc = train(trainloader, model, criterion, optimizer, accelerator, epoch, batch_level_relevance_transforms)
        else:
            train_loss, train_acc = 0.0, 0.0
            accelerator.print("INFO: Skipping training for this epoch as no parameters are trainable.")

        test_loss, test_acc = test(testloader, model, criterion, accelerator)
        
        if scheduler and not args.fixed_lr and optimizer.param_groups and optimizer.param_groups[0]['params']:
             accelerator.wait_for_everyone() 
             if accelerator.is_main_process : 
                if scheduler is not None: scheduler.step()
        
        logger.append([current_lr, train_loss, test_loss, train_acc, test_acc])
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if accelerator.is_main_process:
            save_checkpoint_accelerator({'epoch': epoch + 1, 'acc': test_acc, 'best_acc': best_acc,}, is_best, checkpoint_path=args.checkpoint_path_for_saving, accelerator=accelerator, model_to_save=model, optimizer_to_save=optimizer, scheduler_to_save=scheduler)

    if accelerator.is_main_process:
        logger.close()
        accelerator.print(f'Best Test Acc: {best_acc:.2f}')
        with open(exp_results_dir / "final_best_acc.txt", "w") as f: f.write(f"Best Test Accuracy: {best_acc:.4f}\n")

# --- train, test, save_checkpoint_accelerator functions ---
def train(trainloader, model, criterion, optimizer, accelerator, epoch, batch_relevance_transforms):
    model.train()
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    bar = Bar(f'Training E{epoch+1}', max=len(trainloader))
    for batch_idx, batch_data in enumerate(trainloader):
        if len(batch_data) == 3 and args.dataset == 'imagenet': 
            inputs, targets, _paths = batch_data
        else: 
            inputs, targets = batch_data
        data_time.update(time.time() - end)
        current_inputs = inputs
        if batch_relevance_transforms:
            for transform_op in batch_relevance_transforms:
                current_inputs = transform_op(current_inputs, targets) 
        optimizer.zero_grad()
        outputs = model(current_inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        gathered_outputs = accelerator.gather_for_metrics(outputs)
        gathered_targets = accelerator.gather_for_metrics(targets)
        prec1, prec5 = accuracy(gathered_outputs.data, gathered_targets.data, topk=(1, 5))
        avg_loss = accelerator.gather_for_metrics(loss).mean()
        losses.update(avg_loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0)) 
        top5.update(prec5.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if accelerator.is_main_process:
            bar.suffix  = f'({batch_idx+1}/{len(trainloader)}) D:{data_time.avg:.2f}s B:{batch_time.avg:.2f}s | L:{losses.avg:.4f} | T1:{top1.avg:.2f} T5:{top5.avg:.2f}'
            bar.next()
    if accelerator.is_main_process: bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, accelerator):
    model.eval()
    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    bar = Bar('Testing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            if len(batch_data) == 3 and args.dataset == 'imagenet':
                inputs, targets, _ = batch_data
            else:
                inputs, targets = batch_data
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            gathered_outputs = accelerator.gather_for_metrics(outputs)
            gathered_targets = accelerator.gather_for_metrics(targets)
            avg_loss = accelerator.gather_for_metrics(loss).mean()
            prec1, prec5 = accuracy(gathered_outputs.data, gathered_targets.data, topk=(1, 5))
            losses.update(avg_loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if accelerator.is_main_process:
                bar.suffix = f'({batch_idx+1}/{len(testloader)}) B:{batch_time.avg:.2f}s | L:{losses.avg:.4f} | T1:{top1.avg:.2f} T5:{top5.avg:.2f}'
                bar.next()
    if accelerator.is_main_process: bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint_accelerator(state_dict_content, is_best, checkpoint_path, accelerator, model_to_save, optimizer_to_save, scheduler_to_save=None, filename='checkpoint.pth.tar'):
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model_to_save)
        save_obj = {'epoch': state_dict_content['epoch'], 'state_dict': unwrapped_model.state_dict(), 'acc': state_dict_content['acc'], 'best_acc': state_dict_content['best_acc']}
        if optimizer_to_save.param_groups and optimizer_to_save.param_groups[0]['params']:
            save_obj['optimizer'] = optimizer_to_save.state_dict()
            if scheduler_to_save: 
                save_obj['scheduler'] = scheduler_to_save.state_dict()
        else:
            save_obj['optimizer'] = None 
            save_obj['scheduler'] = None

        filepath = Path(checkpoint_path) / filename
        torch.save(save_obj, filepath)
        if is_best: shutil.copyfile(filepath, Path(checkpoint_path) / 'model_best.pth.tar')

if __name__ == '__main__':
    main()