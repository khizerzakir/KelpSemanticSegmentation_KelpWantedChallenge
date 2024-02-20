import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize
import torch
from sklearn.model_selection import train_test_split
import random

class KelpDataset(Dataset):
    def __init__(self, metadata, data_path, distance_map_path=None, ndvi_path=None, label_path=None, data_transforms=None, label_transforms=None):
        self.metadata = metadata
        self.data_path = data_path
        self.distance_map_path = distance_map_path
        self.ndvi_path = ndvi_path
        self.label_path = label_path
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Load spectral bands and cloud masks
        data_name = os.path.join(self.data_path, self.metadata.iloc[idx]['filename'].replace('.tif', '.npz'))
        with np.load(data_name) as data_file:
            data = data_file[data_file.files[0]]
        spectral_bands = data[[0, 1, 2, 3, 4], :, :]
        cloud_mask = np.expand_dims(data[5, :, :], axis=0)  # Shape is [1, height, width]
        dem = np.expand_dims(data[6, :, :], axis=0)
        
        # Convert from Numpy Arrays to Torch Tensors
        spectral_bands = torch.tensor(spectral_bands, dtype=torch.float32)
        dem = torch.tensor(dem, dtype=torch.float32)
        cloud_mask = torch.tensor(cloud_mask, dtype=torch.uint8)

        # Load distance map
        distance_map_name = os.path.join(self.distance_map_path, self.metadata.iloc[idx]['tile_id'] + '_satellite_distance_map.npz')
        with np.load(distance_map_name) as distance_map_file:
            distance_map = distance_map_file[distance_map_file.files[0]]
        distance_map = np.expand_dims(distance_map, axis=-1)  # Shape is [height, width, 1]
        distance_map = torch.tensor(distance_map, dtype=torch.float32).permute(2, 0, 1)  # Shape is [1, height, width]
        
        # Load ndvi
        ndvi_name = os.path.join(self.ndvi_path, self.metadata.iloc[idx]['tile_id'] + '_satellite.npz')
        with np.load(ndvi_name) as ndvi_file:
            ndvi = ndvi_file[ndvi_file.files[0]]
        ndvi = np.expand_dims(ndvi, axis=-1)  # Shape is [height, width, 1]
        ndvi = torch.tensor(ndvi, dtype=torch.float32).permute(2, 0, 1)  # Shape is [1, height, width]
        
        # Load label
        label = None
        if self.label_path:
            label_name = os.path.join(self.label_path, self.metadata.iloc[idx]['tile_id'] + '_kelp.npz')
            if os.path.exists(label_name):
                with np.load(label_name) as label_file:
                    label = torch.tensor(label_file[label_file.files[0]], dtype=torch.uint8)
        if label is None:
            # Initialize label with zeros if not available
            default_label_shape = (1, data.shape[1], data.shape[2])
            label = torch.zeros(default_label_shape, dtype=torch.uint8)

        # Stack spectral bands, distance map, dem & ndvi
        data = torch.cat([spectral_bands, distance_map, dem, ndvi], dim=0)

        # Apply transformations if provided
        if self.data_transforms is not None:
            data = self.data_transforms(data)
        if self.label_transforms is not None and label is not None:
            label = self.label_transforms(label)

        return data, label, cloud_mask, self.metadata.iloc[idx]['tile_id']

def set_random_seeds(seed_value=44):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def prepare_data(raw_data, processed_data, batch_size, split_mode, bin_count=20, random_state=44):
    
    # Ensure reproducibility
    set_random_seeds(random_state)
                    
    # File paths
    metadata_csv = os.path.join(raw_data, 'metadata_fTq0l2T.csv')
    data_statistics = os.path.join(processed_data, 'train_img_statistics.csv')

    # Npz data paths
    train_satellite_np = os.path.join(processed_data, 'train_satellite_np')
    train_distance_maps_np = os.path.join(processed_data, 'train_distance_maps_np')
    train_ndvi_np = os.path.join(processed_data, 'train_ndvi_np')
    train_kelp_np = os.path.join(processed_data, 'train_kelp_np')
    test_satellite_np = os.path.join(processed_data, 'test_satellite_np')
    test_distance_maps_np =  os.path.join(processed_data, 'test_distance_maps_np')
    test_ndvi_np = os.path.join(processed_data, 'test_ndvi_np')

    # Load and prepare metadata
    metadata = pd.read_csv(metadata_csv)
    train_statistics = pd.read_csv(data_statistics)
    train_metadata = metadata[metadata['in_train'] == True]
    test_given_metadata = metadata[(metadata['in_train'] == False) & (metadata['type'] == 'satellite')]
    train_metadata_stats = pd.merge(train_metadata, train_statistics, right_on='image_id', left_on = 'tile_id', how='inner')
    filtered_train_metadata = train_metadata_stats[(train_metadata_stats['num_kelp_px'] > 0) & (train_metadata_stats['perc_clouds'] < 0.2) & (train_metadata_stats['perc_corrupt'] < 0.1)] 
    
    # Normalization statistics
    all_means= [9.88597260e+03, 1.05878845e+04, 8.53884955e+03, 8.68310337e+03, 8.38589389e+03, 5.40829731e+01, 4.76659551e+00, 7.67201252e-02]
    all_stds= [3.47642640e+03, 4.20390591e+03, 1.32121439e+03, 1.12998422e+03, 7.92201213e+02, 8.30804470e+01, 5.23900800e+02, 1.14605740e-01]
    
    # Transformations
    data_transforms = Compose([
        Normalize(mean=all_means, std=all_stds)
    ])
   
    # Split Train-Val and instantiate datasets and dataloaders
    bin_numbers = pd.qcut(x=filtered_train_metadata['perc_kelp'], q=bin_count, labels=False, duplicates='drop')
    
    if split_mode == 'train_val_test':
        train_metadata, temp_metadata = train_test_split(filtered_train_metadata, test_size=960, random_state=random_state, stratify= bin_numbers)
        bin_numbers_temp = pd.qcut(x=temp_metadata['perc_kelp'], q=bin_count, labels=False, duplicates='drop')  
        val_metadata, test_metadata = train_test_split(temp_metadata, test_size=0.5, random_state=random_state, stratify= bin_numbers_temp)
        
        test_dataset = KelpDataset(metadata=test_metadata, data_path=train_satellite_np, distance_map_path=train_distance_maps_np, ndvi_path=train_ndvi_np, label_path=train_kelp_np, data_transforms=data_transforms, label_transforms=None)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif split_mode == 'train_val':
        train_metadata, val_metadata = train_test_split(filtered_train_metadata, test_size=640, random_state=random_state, stratify=bin_numbers)
        test_dataset = None
   
    train_dataset = KelpDataset(metadata=train_metadata, data_path=train_satellite_np, distance_map_path=train_distance_maps_np, ndvi_path=train_ndvi_np, label_path=train_kelp_np, data_transforms=data_transforms, label_transforms=None)
    val_dataset = KelpDataset(metadata=val_metadata, data_path=train_satellite_np, distance_map_path=train_distance_maps_np, ndvi_path=train_ndvi_np, label_path=train_kelp_np, data_transforms=data_transforms, label_transforms=None)
    test_given_dataset = KelpDataset(metadata=test_given_metadata, data_path=test_satellite_np, distance_map_path=test_distance_maps_np, ndvi_path=test_ndvi_np,  label_path=None, data_transforms=data_transforms, label_transforms=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_given_loader = DataLoader(test_given_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, test_given_loader, all_means, all_stds
   