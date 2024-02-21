import argparse
import torch
from data_preparation import prepare_data
from models import get_model
from train import DiceLoss, train_model, save_learning_curves
from predict import postprocess_and_export_predictions
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

def parse_args():
    parser = argparse.ArgumentParser(description='Kelp Segmentation Training and Prediction')
    parser.add_argument('--raw_data', type=str, required=True, help='Path to the raw data directory')
    parser.add_argument('--processed_data', type=str, required=True, help='Path to the processed data directory')
    parser.add_argument('--outputs', type=str, required=True, help='Path to the outputs directory')
    parser.add_argument('--split_mode', type=str, default='train_val', help='Mode for splitting training dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--use_distance_maps', action='store_true', help='Flag to use distance maps')
    parser.add_argument('--use_dems', action='store_true', help='Flag to use DEMs')
    parser.add_argument('--use_ndvi', action='store_true', help='Flag to use NDVI')
    parser.add_argument('--use_checkpoint', action='store_true', help='Flag to resume training from a checkpoint')
    parser.add_argument('--model', type=str, default='unet_modified', help='Model to use for training')
    parser.add_argument('--loss_func', type=str, default='DiceLoss', help='Loss function for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler for learning rate adjustment')
    parser.add_argument('--gamma', type=float, default=0.95, help='Gamma for ExponentialLR scheduler')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Get the current GPU's total memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"Cuda Total Memory: {total_memory / 1e9} GB")
    
    # Data preparation
    train_loader, val_loader, test_loader, test_given_loader, all_means, all_stds = prepare_data(args.raw_data, args.processed_data, args.batch_size, args.split_mode)    
    print("Data preparation done")
    
    # Model instantiation
    model = get_model(args.model, args.use_distance_maps, args.use_dems, args.use_ndvi)
    print("Model Instantiation done")
    
    # Instantiate loss function
    if args.loss_func == 'DiceLoss':
        loss_func = DiceLoss()
        print("Loss Function instantiation done")
    else:
        raise ValueError(f"Loss function '{args.loss_func}' not supported.")

    # Instantiate optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("Optimizer instantiation done")
    else:
        raise ValueError(f"Optimizer '{args.optimizer}' not supported.")
    
    # Instantiate scheduler if specified
    if args.scheduler and args.gamma:
        if args.scheduler == 'ExponentialLR':
            gamma = float(args.gamma)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = None
    print("Scheduler instantiation done")

    # Training
    model, time_epoch, train_loss_history, val_loss_history, train_iou_history, val_iou_history, best_epoch = train_model(
        model=model, 
        optimizer=optimizer,        
        loss_func=loss_func,         
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device, 
        num_epochs=args.num_epochs,
        outputs=args.outputs,
        use_distance_maps=args.use_distance_maps, 
        use_dems=args.use_dems, 
        use_ndvi=args.use_ndvi,
        use_checkpoint=args.use_checkpoint)
    
    # Saving learning curves and making predictions
    save_learning_curves(train_loss_history, val_loss_history, train_iou_history, val_iou_history, best_epoch, args.outputs)
    
    if args.split_mode == 'train_val_test':
        loader_for_predictions =  test_loader
        
    elif args.split_mode == 'train_val':
        loader_for_predictions = test_given_loader
        
    postprocess_and_export_predictions(
        model=model, 
        test_given_loader=loader_for_predictions,
        device=device, 
        outputs=args.outputs, 
        use_distance_maps=args.use_distance_maps, 
        use_dems=args.use_dems, 
        use_ndvi=args.use_ndvi,
        all_means=all_means, 
        all_stds=all_stds
        )

if __name__ == '__main__':
    main()
