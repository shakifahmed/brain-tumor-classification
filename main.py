import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import ParallelResNetSwin
from dataset import CustomImageDataset
from train import train_model
from evaluate import evaluate_model
from cross_validation import cross_validate_model, cross_validate_report
from visualize import plot_training_history, dataset_images, evaluation_results, cross_validate_results, model_outcomes

if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    label_map = {
        0: 'Glioma',
        1: 'Meningioma',
        2: 'Normal',
        3: 'Pituitary'
    }
    num_epochs = 10
    num_epochs_cv = 2 
    num_folds =  5
    num_classes = 4 
    batch_size = 32

    # Create dataset
    dataset = CustomImageDataset(
        img_dir='your dataset path',
        transform=transform
    )
    
    # show Label Encoding Result
    dataset._show_encoded_labels()
    # show dataset images
    dataset_images(dataset, num_samples=10, columns=5)

    # set DataLoader
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = ParallelResNetSwin(num_classes=num_classes)
    
    # cuda device initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training Phase
    training_history = train_model(model, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, device=device)
    # Plot training history
    plot_training_history(training_history)
    # Model outcomes
    model_outcomes(model, val_dataset, num_samples=10, columns=5, label_map=label_map, device=device)

    # Evaluate the model
    state_dict = torch.load('best_model.pth', map_location=device, weights_only=True) # Load the model 
    model.load_state_dict(state_dict)
    accuracy, precision, recall, f1, true_classes, pred_classes, cm = evaluate_model(model, val_loader, label_map=label_map, device=device)
    evaluation_results(accuracy, precision, recall, f1, true_classes, pred_classes, cm, label_map=label_map)

    # cross validation
    fold_results, all_val_true_labels, all_val_pred_labels = cross_validate_model(model, dataset, num_folds=num_folds, num_epochs=num_epochs_cv, batch_size=batch_size, learning_rate=1e-5, device=device)
    # cross validation report
    df_fold_results, flat_true_labels, flat_pred_labels, tables = cross_validate_report(fold_results, all_val_true_labels, all_val_pred_labels, label_map=label_map, num_folds=num_folds)
    cross_validate_results(df_fold_results, all_val_true_labels, all_val_pred_labels, flat_true_labels, flat_pred_labels, tables, label_map=label_map, num_folds=num_folds)
