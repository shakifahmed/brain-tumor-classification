from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, SubsetRandomSampler
from train import train_model

def cross_validate_model(model, dataset, num_folds=2, num_epochs=4, batch_size=32, learning_rate=1e-3, device=None):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    labels = [label for _, label in dataset.img_labels]

    fold_results = []
    all_val_true_labels = []
    all_val_pred_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.img_labels, labels)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Sample elements randomly from a given list of ids, no replacement.
        train_dataset = SubsetRandomSampler(train_idx)
        val_dataset = SubsetRandomSampler(val_idx)
        
        # Define data loaders for training and validation data in this fold
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_dataset, num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_dataset, num_workers=4, pin_memory=True)
        # Initialize a new model instance for each fold
        fold_model = type(model)().to(device)  # Create a new instance of the same model type
        fold_model.load_state_dict(model.state_dict()) 
        # Train the model
        fold_result = train_model(fold_model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, cross_val=True, device=device)
        
        fold_results.append({
            'fold': fold + 1,
            'train_loss': fold_result['train_loss'],
            'train_accuracy': fold_result['train_accuracy'],
            'val_loss': fold_result['val_loss'],
            'val_accuracy': fold_result['val_accuracy']
        })
        
        all_val_true_labels.append(fold_result['val_true_labels'])
        all_val_pred_labels.append(fold_result['val_pred_labels'])

    return fold_results, all_val_true_labels, all_val_pred_labels

def cross_validate_report(fold_results, all_val_true_labels, all_val_pred_labels, label_map=None, num_folds=2):
    # Create a DataFrame with fold results
    df_fold_results = pd.DataFrame(fold_results)
    # Flatten the nested lists of labels for overall metrics
    flat_true_labels = np.concatenate(all_val_true_labels)
    flat_pred_labels = np.concatenate(all_val_pred_labels)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    class_names = list(label_map.values())
    results = {metric: {class_name: [] for class_name in class_names} for metric in metrics}
    
    for fold in range(num_folds):
        true_labels = [label_map[label] for label in all_val_true_labels[fold]]
        pred_labels = [label_map[label] for label in all_val_pred_labels[fold]]
        
        # Calculate metrics for this fold
        report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
        for class_name in class_names:
            results['Accuracy'][class_name].append(report[class_name]['precision'])  # Accuracy per class is the same as precision
            results['Precision'][class_name].append(report[class_name]['precision'])
            results['Recall'][class_name].append(report[class_name]['recall'])
            results['F1-Score'][class_name].append(report[class_name]['f1-score'])
    
    # Calculate mean and variance
    for metric in metrics:
        for class_name in class_names:
            values = results[metric][class_name]
            results[metric][class_name] = {
                'folds': values,
                'mean': np.mean(values),
                'variance': np.var(values)
            }
    
    # Generate tables
    tables = []
    for metric in metrics:
        table_data = []
        for class_name in class_names:
            row = [class_name] + results[metric][class_name]['folds'] + [
                results[metric][class_name]['mean'],
                results[metric][class_name]['variance']
            ]
            table_data.append(row)
        
        headers = ['Class'] + [f'Fold {i+1}' for i in range(num_folds)] + ['μ', 'σ²']
        table = tabulate(table_data, headers=headers, floatfmt='.2f', tablefmt='grid')
        tables.append(f"\n{metric}\n{table}")
    tables = '\n'.join(tables)

    return df_fold_results, flat_true_labels, flat_pred_labels, tables