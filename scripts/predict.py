import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import tarfile

def postprocess_and_export_predictions(model, test_given_loader, device, outputs, use_distance_maps, use_dems, use_ndvi, all_means, all_stds, postprocess=True, threshold=0, p_threshold=0.5):
    model.to(device)
    model.eval()
    
    # Set up predictions path
    predictions_path =  os.path.join(outputs, 'predictions')
    
    # Ensure the predictions directory exists
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    # Determine indices to drop
    drop_indices = []
    if not use_distance_maps:
        drop_indices.append(5)  # Index 5 is for distance maps
    if not use_dems:
        drop_indices.append(6)  # Index 6 is for DEMs
    if not use_ndvi:
        drop_indices.append(7)  # Index 7 is for NDVI

    with torch.no_grad():
        for data_batch, _, _, tile_ids_batch in tqdm(test_given_loader, desc='Processing', leave=True):
            # Extract the DEM band
            dem_band_batch = data_batch[:, 6:7, :, :]

            # Unnormalize the DEM band using the mean and standard deviation
            if all_means is not None and all_stds is not None:
                dem_mean = all_means[6]
                dem_std = all_stds[6]
                dem_band_batch = (dem_band_batch * dem_std) + dem_mean

            # Extract the red band for additional verification
            red_band_batch = data_batch[:, 2:3, :, :]

            # Drop bands not included in training
            if drop_indices:
                data_batch = data_batch[:, [i for i in range(data_batch.shape[1]) if i not in drop_indices], :, :]

            data_batch = data_batch.to(device)
            outputs = model(data_batch)
            predictions = torch.sigmoid(outputs) > p_threshold

            for idx, prediction in enumerate(predictions):
                # Verification based on the red band
                invalid_mask = red_band_batch[idx] == -32768

                if postprocess:
                    # Use the previously extracted DEM band for post-processing
                    dem_band = dem_band_batch[idx].to(device)

                    # Create a mask where DEM values are greater than the threshold
                    mask = dem_band > threshold

                    # Adjust prediction tensor shape if necessary
                    if prediction.dim() > mask.dim():
                        prediction = prediction.squeeze(0)

                    prediction[mask] = 0  # Apply the mask to the prediction

                # Set invalid pixels to 0 based on the red band verification
                prediction[invalid_mask] = 0

                # Prepare the prediction for saving
                prediction = prediction.cpu().squeeze().numpy().astype(np.uint8)

                # Save the prediction as a TIFF file
                tiff_filename = os.path.join(predictions_path, f'{tile_ids_batch[idx]}_kelp.tif')
                Image.fromarray(prediction).save(tiff_filename, format='TIFF')

    # Create an archive of all the prediction TIFF files
    archive_name = 'predictions.tar.gz'
    with tarfile.open(os.path.join(predictions_path, archive_name), 'w:gz') as tar:
        for file in os.listdir(predictions_path):
            if file.endswith('.tif'):
                tar.add(os.path.join(predictions_path, file), arcname=file)

    print(f'All predictions are saved and archived in {predictions_path}, archive name: {archive_name}')