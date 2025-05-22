import numpy as np
import torch
import yaml
from zennit.attribution import Gradient
from zennit.rules import ZPlus, Epsilon, Pass, Flat
from zennit.canonizers import NamedMergeBatchNorm
from zennit.composites import NameMapComposite

def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_cannonizer_map(file_path):
    with open(file_path, 'r') as f:
        return [[line.strip().split(', ')[0:1], line.strip().split(', ')[1]] for line in f if line.strip()]

def load_name_map(file_path):
    with open(file_path, 'r') as f:
        return [([line.strip().split(', ')[0]], eval(line.strip().split(', ')[1])) for line in f if line.strip()]


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875, part_list=None, epoch=None, dir=None):
    """ batch_pc: BxNx3 """
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
            
            if part_list is not None and dir is not None:
                import os
                FINAL_PATH = os.path.join(dir, part_list[b])
                if not os.path.exists(FINAL_PATH):
                    os.makedirs(FINAL_PATH)
                    
                DROP_IDX_NAME_EPOCH = part_list[b] + '_' + str(epoch)
                DROP_IDX_FINAL_DIR = os.path.join(FINAL_PATH, DROP_IDX_NAME_EPOCH)
                np.save(DROP_IDX_FINAL_DIR, drop_idx)
    return batch_pc

def relavance_point_dropout(batch_pc, target, alpha, classifier, max_dropout_ratio, num_class, epoch, part_list, dir, name_map_path, cannonizer_map_path, replacement_value):
    """ Dropout points based on LRP relevance scores """
    # Load NAME_MAP and CANNONNIZER_MAP from external YAML files
    # name_map_config = load_yaml_config(name_map_path)
    # # cannonizer_map_config = load_yaml_config(cannonizer_map_path)
    # CANNONIZER_MAP = load_cannonizer_map(cannonizer_map_path)

    # NAME_MAP = [(entry[0], eval(entry[1])) for entry in name_map_config['name_map']]
    # # CANNONNIZER_MAP = cannonizer_map_config['cannonizer_map']
    # print("CANNONNIZER_MAP:", CANNONIZER_MAP)
    
    # Load NAME_MAP and CANNONIZER_MAP from external text files
    # NAME_MAP = load_name_map(name_map_path)
    # CANNONIZER_MAP = load_cannonizer_map(cannonizer_map_path)
    
    NAME_MAP_MSG = [
    (['sa1.conv_blocks.0.0'], Flat()),  # Conv2d
    (['sa1.conv_blocks.0.1'], Epsilon()),  # Conv2d
    (['sa1.conv_blocks.0.2'], Epsilon()),  # Conv2d
    (['sa1.conv_blocks.1.0'], Epsilon()),  # Conv2d
    (['sa1.conv_blocks.1.1'], Epsilon()),  # Conv2d
    (['sa1.conv_blocks.1.2'], Epsilon()),  # Conv2d
    (['sa1.conv_blocks.2.0'], Epsilon()),  # Conv2d
    (['sa1.conv_blocks.2.1'], Epsilon()),  # Conv2d
    (['sa1.conv_blocks.2.2'], Epsilon()),  # Conv2d
    (['sa1.bn_blocks.0.0'], Pass()),  # BatchNorm2d
    (['sa1.bn_blocks.0.1'], Pass()),  # BatchNorm2d
    (['sa1.bn_blocks.0.2'], Pass()),  # BatchNorm2d
    (['sa1.bn_blocks.1.0'], Pass()),  # BatchNorm2d
    (['sa1.bn_blocks.1.1'], Pass()),  # BatchNorm2d
    (['sa1.bn_blocks.1.2'], Pass()),  # BatchNorm2d
    (['sa1.bn_blocks.2.0'], Pass()),  # BatchNorm2d
    (['sa1.bn_blocks.2.1'], Pass()),  # BatchNorm2d
    (['sa1.bn_blocks.2.2'], Pass()),  # BatchNorm2d
    (['sa2.conv_blocks.0.0'], Epsilon()),  # Conv2d
    (['sa2.conv_blocks.0.1'], Epsilon()),  # Conv2d
    (['sa2.conv_blocks.0.2'], Epsilon()),  # Conv2d
    (['sa2.conv_blocks.1.0'], Epsilon()),  # Conv2d
    (['sa2.conv_blocks.1.1'], Epsilon()),  # Conv2d
    (['sa2.conv_blocks.1.2'], Epsilon()),  # Conv2d
    (['sa2.conv_blocks.2.0'], Epsilon()),  # Conv2d
    (['sa2.conv_blocks.2.1'], Epsilon()),  # Conv2d
    (['sa2.conv_blocks.2.2'], Epsilon()),  # Conv2d
    (['sa2.bn_blocks.0.0'], Pass()),  # BatchNorm2d
    (['sa2.bn_blocks.0.1'], Pass()),  # BatchNorm2d
    (['sa2.bn_blocks.0.2'], Pass()),  # BatchNorm2d
    (['sa2.bn_blocks.1.0'], Pass()),  # BatchNorm2d
    (['sa2.bn_blocks.1.1'], Pass()),  # BatchNorm2d
    (['sa2.bn_blocks.1.2'], Pass()),  # BatchNorm2d
    (['sa2.bn_blocks.2.0'], Pass()),  # BatchNorm2d
    (['sa2.bn_blocks.2.1'], Pass()),  # BatchNorm2d
    (['sa2.bn_blocks.2.2'], Pass()),  # BatchNorm2d
    (['sa3.mlp_convs.0'], Epsilon()),  # Conv2d
    (['sa3.mlp_convs.1'], Epsilon()),  # Conv2d
    (['sa3.mlp_convs.2'], Epsilon()),  # Conv2d
    (['sa3.mlp_bns.0'], Pass()),  # BatchNorm2d
    (['sa3.mlp_bns.1'], Pass()),  # BatchNorm2d
    (['sa3.mlp_bns.2'], Pass()),  # BatchNorm2d
    (['fc1'], Epsilon()),  # Linear
    (['bn1'], Pass()),  # BatchNorm1d
    (['drop1'], Pass()),  # Dropout
    (['fc2'], Epsilon()),  # Linear
    (['bn2'], Pass()),  # BatchNorm1d
    (['drop2'], Pass()),  # Dropout
    (['fc3'], Epsilon()),  # Linear
    ]
        
    CANNONNIZER_MAP_MSG = [
    [["sa1.conv_blocks.0.0"], "sa1.bn_blocks.0.0"],
    [["sa1.conv_blocks.0.1"], "sa1.bn_blocks.0.1"],
    [["sa1.conv_blocks.0.2"], "sa1.bn_blocks.0.2"],
    [["sa1.conv_blocks.1.0"], "sa1.bn_blocks.1.0"],
    [["sa1.conv_blocks.1.1"], "sa1.bn_blocks.1.1"],
    [["sa1.conv_blocks.1.2"], "sa1.bn_blocks.1.2"],
    [["sa1.conv_blocks.2.0"], "sa1.bn_blocks.2.0"],
    [["sa1.conv_blocks.2.1"], "sa1.bn_blocks.2.1"],
    [["sa1.conv_blocks.2.2"], "sa1.bn_blocks.2.2"],
    [["sa2.conv_blocks.0.0"], "sa2.bn_blocks.0.0"],
    [["sa2.conv_blocks.0.1"], "sa2.bn_blocks.0.1"],
    [["sa2.conv_blocks.0.2"], "sa2.bn_blocks.0.2"],
    [["sa2.conv_blocks.1.0"], "sa2.bn_blocks.1.0"],
    [["sa2.conv_blocks.1.1"], "sa2.bn_blocks.1.1"],
    [["sa2.conv_blocks.1.2"], "sa2.bn_blocks.1.2"],
    [["sa2.conv_blocks.2.0"], "sa2.bn_blocks.2.0"],
    [["sa2.conv_blocks.2.1"], "sa2.bn_blocks.2.1"],
    [["sa2.conv_blocks.2.2"], "sa2.bn_blocks.2.2"],
    [["sa3.mlp_convs.0"], "sa3.mlp_bns.0"],
    [["sa3.mlp_convs.1"], "sa3.mlp_bns.1"],
    [["sa3.mlp_convs.2"], "sa3.mlp_bns.2"],
    [["fc1"], "bn1"],
    [["fc2"], "bn2"]
]
    
    CANNONNIZER = NamedMergeBatchNorm(CANNONNIZER_MAP_MSG)
    COMPOSITE = NameMapComposite(name_map=NAME_MAP_MSG, canonizers=[CANNONNIZER])
    
    # Ensure batch_pc has requires_grad=True
    batch_pc = torch.Tensor(batch_pc).cuda().requires_grad_()
    target = torch.Tensor(target).long().cuda()
    batch_pc = batch_pc.transpose(2, 1)

    # Ensure TARGET is on the same device as batch_pc
    TARGET = torch.eye(num_class, device=batch_pc.device)[target]

    with Gradient(model=classifier.eval(), composite=COMPOSITE) as ATTRIBUTOR:
        output, attribution = ATTRIBUTOR(batch_pc, TARGET)
        relevance = torch.squeeze(attribution.sum(1))

        for b in range(batch_pc.shape[0]):
            relevance[b] = (relevance[b] - relevance[b].min()) / (relevance[b].max() - relevance[b].min())
            relevance_np = relevance[b].cpu().detach().numpy()
            drop_idx = np.where(((alpha * np.random.random((batch_pc.shape[2]))) + 
                               ((1.0 - alpha) * (relevance_np))) >= (1.0 - max_dropout_ratio))[0]
            
            if len(drop_idx) > 0:
                batch_pc_np = batch_pc.detach().cpu().numpy()
                batch_pc_np[b, :, drop_idx] = replacement_value  # Set dropped points to replacement value
                batch_pc = torch.Tensor(batch_pc_np).cuda()

    batch_pc = batch_pc.transpose(1, 2)
    return batch_pc.cpu().detach().numpy()

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, C))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data
