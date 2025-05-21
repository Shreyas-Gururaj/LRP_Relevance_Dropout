import os
import sys
import torch
import numpy as np
import logging
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import PartNormalDataset
from models import pointnet_cls, pointnet2_cls_ssg, pointnet2_cls_msg
from utils import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', type=str, default="/home/fe/gururaj/LRP_Experiment/lrp_relevance_dropout-shreyas/3DPC/config/train_config.yaml", help='Path to the training configuration file')
    parser.add_argument('--name_map', type=str, default="/home/fe/gururaj/LRP_Experiment/lrp_relevance_dropout-shreyas/3DPC/config/name_map.txt", help='Path to the NAME_MAP configuration file')
    parser.add_argument('--cannonizer_map', type=str, default="/home/fe/gururaj/LRP_Experiment/lrp_relevance_dropout-shreyas/3DPC/config/canonizer_map.txt", help='Path to the CANNONIZER_MAP configuration file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test(model, loader, num_class, device):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target, full_path) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.to(device), target.to(device)
        points = points.transpose(2, 1)
        # Forward pass
        pred = classifier(points)  # Adjusted for models that return only one value
        if isinstance(pred, tuple):
            pred, trans_feat = pred  # Unpack only if it's a tuple
        else:
            trans_feat = None  # Handle the case where trans_feat is not returned
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc

def main():
    args = parse_args()
    config = load_config(args.config)

    # Set up logging
    log_dir = Path(config['training']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
        logging.FileHandler(log_dir / 'training.log'),
        logging.StreamHandler()
    ])
    logger = logging.getLogger()

    # Load dataset
    logger.info('Loading dataset...')
    dataset_name = config['dataset']['name']
    if dataset_name == 'ModelNet40':
        train_dataset = ModelNetDataLoader(
            root=config['dataset']['data_path'],
            args=config['dataset'],
            split='train',
            process_data=config['dataset']['process_data']
        )
        test_dataset = ModelNetDataLoader(
            root=config['dataset']['data_path'],
            args=config['dataset'],
            split='test',
            process_data=config['dataset']['process_data']
        )
    elif dataset_name == 'ShapeNet':
        train_dataset = PartNormalDataset(
            root=config['dataset']['data_path'],
            npoints=config['dataset']['num_points'],
            split='trainval',
            normal_channel=config['dataset']['use_normals']
        )
        test_dataset = PartNormalDataset(
            root=config['dataset']['data_path'],
            npoints=config['dataset']['num_points'],
            split='test',
            normal_channel=config['dataset']['use_normals']
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=10, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=10)

    # Load model
    logger.info('Loading model...')
    model_name = config['model']['name']
    if model_name == 'pointnet_cls':
        model = pointnet_cls.get_model(config['model']['num_class'], normal_channel=config['model']['normal_channel'])
        criterion = pointnet_cls.get_loss()
    elif model_name == 'pointnet2_cls_ssg':
        model = pointnet2_cls_ssg.get_model(config['model']['num_class'], normal_channel=config['model']['normal_channel'])
        criterion = pointnet2_cls_ssg.get_loss()
    elif model_name == 'pointnet2_cls_msg':
        model = pointnet2_cls_msg.get_model(config['model']['num_class'], normal_channel=config['model']['normal_channel'])
        criterion = pointnet2_cls_msg.get_loss()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # Training loop
    logger.info('Starting training...')
    best_instance_acc = 0.0
    best_class_acc = 0.0

    for epoch in range(config['training']['epochs']):
        logger.info(f'Epoch {epoch + 1}/{config["training"]["epochs"]}')
        model.train()
        mean_correct = []

        for points, target, full_path in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            file_name_list = [os.path.splitext(os.path.basename(fp))[0] for fp in full_path]

            # Convert to numpy for augmentations
            points = points.data.numpy()

            # Apply relevance-based dropout
            points = provider.relavance_point_dropout(
                batch_pc=points,
                target=target,
                alpha=config['augmentation']['relevance_dropout']['alpha'],
                classifier=model.eval(),
                max_dropout_ratio=config['augmentation']['relevance_dropout']['beta'],
                num_class=config['model']['num_class'],
                epoch=epoch,
                part_list=file_name_list,
                dir=config['logging']['save_dir'],
                name_map_path=args.name_map,
                cannonizer_map_path=args.cannonizer_map,
                replacement_value=config['augmentation']['relevance_dropout']['value_replace']
            )

            # Apply other augmentations
            if config['augmentation']['scaling']['enabled']:
                points[:, :, 0:3] = provider.random_scale_point_cloud(
                    points[:, :, 0:3],
                    scale_low=config['augmentation']['scaling']['scale_low'],
                    scale_high=config['augmentation']['scaling']['scale_high']
                )
            if config['augmentation']['shifting']['enabled']:
                points[:, :, 0:3] = provider.shift_point_cloud(
                    points[:, :, 0:3],
                    shift_range=config['augmentation']['shifting']['shift_range']
                )

            # Convert back to tensor and transpose
            points = torch.Tensor(points).transpose(2, 1).to(device)
            target = target.to(device)

            # Forward pass
            pred = model(points)  # Adjusted for models that return only one value
            if isinstance(pred, tuple):
                pred, trans_feat = pred  # Unpack only if it's a tuple
            else:
                trans_feat = None  # Handle the case where trans_feat is not returned
            loss = criterion(pred, target.long(), trans_feat)
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        train_instance_acc = np.mean(mean_correct)
        logger.info(f'Train Instance Accuracy: {train_instance_acc:.4f}')

        # Test
        with torch.no_grad():
            instance_acc, class_acc = test(model.eval(), test_loader, config['model']['num_class'], device)
            logger.info(f'Test Instance Accuracy: {instance_acc:.4f}, Class Accuracy: {class_acc:.4f}')

            if instance_acc > best_instance_acc:
                best_instance_acc = instance_acc
                torch.save(model.state_dict(), log_dir / 'best_model.pth')
                logger.info('Saved best model.')

            if class_acc > best_class_acc:
                best_class_acc = class_acc

    logger.info(f'Best Instance Accuracy: {best_instance_acc:.4f}, Best Class Accuracy: {best_class_acc:.4f}')

if __name__ == '__main__':
    main()
