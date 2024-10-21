import random
import math
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

def dataset_images(dataset, num_samples=20, columns=2):
    random_indices = random.sample(range(len(dataset)), num_samples)
    random_samples = [dataset[idx] for idx in random_indices]

    # Compute the number of rows and columns
    rows = math.ceil(num_samples / columns)
    fig_width = columns * 4  # Adjust width per column
    fig_height = rows * 4    # Adjust height per row

    # Plot the images with their labels
    plt.figure(figsize=(fig_width, fig_height))
    for i, (image, label) in enumerate(random_samples):
        plt.subplot(rows, columns, i + 1)
        image = image.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Unnormalize
        image = image.numpy()
        plt.imshow(image)
        plt.title(f'Label: {label}', fontsize=20)
        plt.axis('off')

    plt.savefig('Images/dataset_images.png')
    plt.close()

def evaluation_results(accuracy, precision, recall, f1, true_classes, pred_classes, cm, label_map=None):
    print('Evaluation Metrics:')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_classes, pred_classes))

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values(), cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Validation Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()
    plt.savefig('Images/confusion_matrix.png')
    plt.close()

def model_outcomes(model, val_dataset, num_samples=10, columns=2, label_map=None, device=None):
    # Randomly select images from the validation dataset
    random_indices = random.sample(range(len(val_dataset)), num_samples)
    random_samples = [val_dataset[idx] for idx in random_indices]

    # Compute the number of rows and columns
    rows = math.ceil(num_samples / columns)
    fig_width = columns * 4  # Adjust width per column
    fig_height = rows * 4    # Adjust height per row

    # Predict the class of the selected images
    predictions = []
    true_labels = []
    images = []
    for image, label, in random_samples:
        images.append(image)
        true_labels.append(label)
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            pred_label = torch.argmax(output, dim=1).item()
        predictions.append(pred_label)

    # Plot the images with their true and predicted labels
    plt.figure(figsize=(fig_width, fig_height))
    for i, (image, true_label, pred_label) in enumerate(zip(images, true_labels, predictions)):
        plt.subplot(rows, columns, i + 1)
        image = image.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Unnormalize
        image = image.numpy()
        plt.imshow(image)
        # Use the label_map to get class names
        true_class = label_map[true_label]
        pred_class = label_map[pred_label]
        
        plt.title(f'True: {true_class}, \nPred: {pred_class}', fontsize=16)
        plt.axis('off')

    plt.savefig('Images/model_outcomes.png')
    plt.close()

def plot_training_history(training_history):
    train_accuracies = training_history['train_accuracies']
    val_accuracies = training_history['val_accuracies']
    train_losses = training_history['train_losses']
    val_losses = training_history['val_losses']

    # Plot training and validation accuracy
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

#     plt.tight_layout()
#     plt.show()
    plt.savefig('Images/plot_training_history.png')
    plt.close()
    
def cross_validate_results(df_fold_results, all_val_true_labels, all_val_pred_labels, flat_true_labels, flat_pred_labels, tables, label_map=None, num_folds=2):
    print("\nResults by fold:\n")
    print(df_fold_results.to_string(index=False)) 
    
    print("\nOverall Classification Report:\n")
    print(classification_report(flat_true_labels, flat_pred_labels, target_names=list(label_map.values())))
    
    # Generate tables
    print('\nCross-Validation Table:')
    print(tables)
    
    # confusion metrics create
    unique_labels = np.unique(flat_true_labels)
    
    # Calculate rows and columns for subplots
    total_plots = num_folds + 1  # Include space for overall confusion matrix
    columns = 3  # Set fixed number of columns (or adjust as needed)
    rows = math.ceil(total_plots / columns)
    
    # Dynamically adjust the figure size
    fig_width = columns * 10
    fig_height = rows * 10
    
    # Create subplots
    fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height))
    fig.suptitle('Confusion Matrices for Each Fold and Overall', fontsize=16)

    # Flatten axes for easy indexing (in case of uneven rows/columns)
    axes = axes.flatten()

    # Plot confusion matrices for each fold
    for fold in range(num_folds):
        true_classes = all_val_true_labels[fold]
        pred_classes = all_val_pred_labels[fold]
        
        # Generate confusion matrix for the current fold
        cm = confusion_matrix(true_classes, pred_classes, labels=unique_labels)

        # Plot confusion matrix for this fold
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values(), ax=axes[fold], cbar=False)
        axes[fold].set_title(f'Fold {fold+1}')
        axes[fold].set_xlabel('Predicted Label')
        axes[fold].set_ylabel('True Label')

        # Rotate x-axis labels for better readability
        axes[fold].set_xticklabels(axes[fold].get_xticklabels(), rotation=45, ha='right')

    # Generate overall confusion matrix
    overall_cm = confusion_matrix(flat_true_labels, flat_pred_labels, labels=unique_labels)

    # Plot overall confusion matrix in the last subplot
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values(), ax=axes[-1], cbar=False)
    axes[-1].set_title('Overall')
    axes[-1].set_xlabel('Predicted Label')
    axes[-1].set_ylabel('True Label')
    
    # Rotate x-axis labels for better readability
    axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=45, ha='right')

    # Hide any unused subplots if there are fewer plots than subplots
    for i in range(total_plots, len(axes)):
        fig.delaxes(axes[i])
    
    # plt.tight_layout()
    # plt.show()
    # Save the figure
    plt.savefig('Images/cross_val_confusion_metrics.png')
    plt.close()