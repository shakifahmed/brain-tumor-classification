
import pickle
import torch
from tqdm import tqdm


# Simple EarlyStopping Class
class EarlyStopping:
    def __init__(self, patience=5, filename='model.pth'):
        self.patience = patience  
        self.best_loss = float('inf')  
        self.counter = 0  
        self.filename = filename 

    def check(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss  
            self.counter = 0 
            torch.save(model.state_dict(), self.filename)  # Save best model
            print(f"Improved validation loss! Model saved at {self.filename}")
        else:
            self.counter += 1  # Increment counter if no improvement
            print(f"No improvement, patience count: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("Stopping early!")
                return True  # Stop training
        return False  # Continue training

def train_model(model, train_loader, val_loader, num_epochs=2, learning_rate=1e-5, cross_val=False, device=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.StepLR(optimizer, step_size=5, gamma=0.1)
    best_model_path = 'best_model.pth' if not cross_val else 'best_model_cross_val.pth'
    early_stopping = EarlyStopping(patience=5, filename=best_model_path)
    model.to(device)
    # best_val_accuracy = 0.0 
    best_val_loss = float('inf')
    
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_true_labels = []
        train_pred_labels = []
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(predicted.cpu().numpy())

        training_loss = train_loss / len(train_loader)
        training_accuracy = correct / total
        train_accuracies.append(training_accuracy)
        train_losses.append(training_loss)

        # Validation
        model.eval()
        val_loss =0.0 
        correct = 0 
        total = 0
        
        val_true_labels = []
        val_pred_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                with torch.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # for cross validation
                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(predicted.cpu().numpy())

        validation_loss = val_loss / len(val_loader)
        validation_accuracy = correct / total
        val_accuracies.append(validation_accuracy)
        val_losses.append(validation_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, train_loss: {training_loss:.4f}, train_accuracy: {training_accuracy:.4f}, val_loss: {validation_loss:.4f}, val_accuracy: {validation_accuracy:.4f}")

        # if validation_accuracy > best_val_accuracy:
        #     best_val_accuracy = validation_accuracy
        #     torch.save(model.state_dict(), best_model_path)

        # if validation_loss < best_val_loss:
        #     best_val_loss = validation_loss
        #     torch.save(model.state_dict(), best_model_path)
        
        # Check if early stopping is needed
        if early_stopping.check(validation_loss, model):
            break

        # scheduler.step()
        
    # Load the model state dict
    state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    # Load the state dict into your model
    model.load_state_dict(state_dict)
   
    # print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best validation loss: {early_stopping.best_loss:.4f}")

    if cross_val:
        return {
            'train_loss': training_loss,
            'train_accuracy': training_accuracy,
            'val_loss': validation_loss,
            'val_accuracy': validation_accuracy,
            'val_true_labels': val_true_labels,
            'val_pred_labels': val_pred_labels
        }
    else:
        training_history = {
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        # Save the dictionary as a pickle file
        with open('training_history.pickle', 'wb') as handle:
            pickle.dump(training_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return training_history