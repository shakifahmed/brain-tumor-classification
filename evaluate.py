import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, val_loader, label_map=None, device=None):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            with torch.autocast(device_type='cuda'):
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    true_classes = [label_map[label] for label in all_labels]
    pred_classes = [label_map[pred] for pred in all_predictions]
    
    cm = confusion_matrix(true_classes, pred_classes, labels=list(label_map.values()))
    accuracy = accuracy_score(true_classes, pred_classes)
    precision = precision_score(true_classes, pred_classes, average='weighted')
    recall = recall_score(true_classes, pred_classes, average='weighted')
    f1 = f1_score(true_classes, pred_classes, average='weighted')

    return accuracy, precision, recall, f1, true_classes, pred_classes, cm