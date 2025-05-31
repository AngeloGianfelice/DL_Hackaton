import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from source.loadData import GraphDataset
from source.utils import set_seed
from source.models import GNN
from source.utils import get_dataset
import argparse

# Set the random seed
set_seed(42)

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, criterion, device, save_checkpoints=False, checkpoint_path=None, current_epoch=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in tqdm(data_loader, desc="Training", unit="batch"):
        data = data.to(device)
        outputs = model(data)
        
        loss = criterion(outputs, data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == data.y).sum().item()
        total += data.y.size(0)

    # Save checkpoint
    if save_checkpoints and checkpoint_path is not None:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader), correct / total

def train_co_teaching(data_loader, model1, model2, optimizer1, optimizer2, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model1.train()
    model2.train()
    total_loss1 = 0
    correct1 = 0
    total1 = 0
    
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        output1 = model1(data)
        output2 = model2(data)
        loss1,loss2 = criterion(output1, output2, data.y)
        
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        
        total_loss1 += loss1.item()
        pred1 = output1.argmax(dim=1)
        correct1 += (pred1 == data.y).sum().item()
        total1 += data.y.size(0)
        
    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model1.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss1 / len(data_loader), correct1 / total1

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(data.y.cpu().numpy())
            else:
                predictions.extend(pred.cpu().numpy())
                
    if calculate_accuracy:
        accuracy = correct / total
        f1 = f1_score(true_labels, predictions, average='macro')
        return  total_loss / len(data_loader),accuracy,f1
        
    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.getcwd() 
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

# NEW NoisyCE
class NoisyCrossEntropyLoss_NEW(nn.Module):
    def __init__(self, p_noisy=0.2):
        super().__init__()
        self.p_noisy = p_noisy
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        true_class_probs = probs[torch.arange(len(targets)), targets]
        uncertainty = 1.0 - true_class_probs.detach()
        weights = (1.0 - self.p_noisy) + self.p_noisy * uncertainty
        weighted_loss = (losses * weights).mean()
        return weighted_loss

# OLD NoisyCE   
class NoisyCrossEntropyLoss_OLD(nn.Module): 
    def __init__(self, p_noisy=0.2): 
        super().__init__() 
        self.p = p_noisy 
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()

class co_teaching_loss(torch.nn.Module):
    def __init__(self, forget_rate=0.3, label_smoothing=0.1):
        super().__init__()
        self.forget_rate = forget_rate
        self.ce = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, logits1, logits2, target):
        loss1 = self.ce(logits1, target)
        loss2 = self.ce(logits2, target)

        # Sort the samples by loss value (ascending)
        ind1_sorted = torch.argsort(loss1)
        ind2_sorted = torch.argsort(loss2)

        # Calculate number of samples to remember
        remember_rate = 1 - self.forget_rate
        num_remember = int(remember_rate * len(loss1))

        # Select indices of the smallest losses (i.e., most confident predictions)
        ind1_update = ind1_sorted[:num_remember]
        ind2_update = ind2_sorted[:num_remember]

        # Exchange and compute updated loss
        loss1_update = self.ce(logits1[ind2_update], target[ind2_update])
        loss2_update = self.ce(logits2[ind1_update], target[ind1_update])

        return torch.mean(loss1_update), torch.mean(loss2_update)


def main(args):

    config = {

        "seed": 42,
        "num_checkpoints": 10,
        "gnn": "gin-virtual",
        "batch_size": 32,
        "forget_rate": 0.3,  # for Co-teaching
        "label_smoothing": 0.1,  # for Co-teaching
        "noise_prob": 0.2,  # for Noisy Losses
        
        "A": {  
            "drop_ratio": 0.4,
            "num_layer": 3,
            "emb_dim": 512,
            "epochs": 100,
            "loss": 2,  # 1 Noisy Old, 2 Noisy New, 3 Co-teaching
            "graph_pooling": "mean",  # mean, max, attention, set2set
        },
        "B": {  
            "drop_ratio": 0.5,
            "num_layer": 5,
            "emb_dim": 300,
            "epochs": 100,
            "loss": 3,  # 1 Noisy Old, 2 Noisy New, 3 Co-teaching
            "graph_pooling": "mean",  # mean, max, attention, set2set
        },
        "C": {  
            "drop_ratio": 0.5,
            "num_layer": 3,
            "emb_dim": 1024,
            "epochs": 100,
            "loss": 1,  # 1 Noisy Old, 2 Noisy New, 3 Co-teaching
            "graph_pooling": "attention",  # mean, max, attention, set2set
        },
        "D": { 
            "drop_ratio": 0.5,
            "num_layer": 5,
            "emb_dim": 256,
            "epochs": 100,
            "loss": 2,  # 1 Noisy Old, 2 Noisy New, 3 Co-teaching
            "graph_pooling": "mean",  # mean, max, attention, set2set
        },
    }
    
    curr_dataset = get_dataset(args.test_path)

    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = config["num_checkpoints"] 
    
    if config["gnn"] == 'gin-virtual':
        model1 = GNN(gnn_type='gin', num_class=6, num_layer=config[curr_dataset]["num_layer"], emb_dim=config[curr_dataset]["emb_dim"], drop_ratio=config[curr_dataset]["drop_ratio"], virtual_node=True, graph_pooling=config[curr_dataset]["graph_pooling"]).to(device)
        if config[curr_dataset]["loss"] == 3:
            model2 = GNN(gnn_type='gin', num_class=6, num_layer=config[curr_dataset]["num_layer"], emb_dim=config[curr_dataset]["emb_dim"], drop_ratio=config[curr_dataset]["drop_ratio"], virtual_node=True, graph_pooling=config[curr_dataset]["graph_pooling"]).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)

    if config[curr_dataset]["loss"] == 1:
        criterion = NoisyCrossEntropyLoss_OLD(config["noise_prob"])
    elif config[curr_dataset]["loss"] == 2:
        criterion = NoisyCrossEntropyLoss_NEW(config["noise_prob"])
    elif config[curr_dataset]["loss"] == 3:
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        criterion = co_teaching_loss(config["forget_rate"],config["label_smoothing"])
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if args.train_path:
        # Clear existing logging handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=log_file,  filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())

    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)


    if os.path.exists(checkpoint_path) and not args.train_path:
        model1.load_state_dict(torch.load(checkpoint_path,map_location=device))
        print(f"Loaded best model from {checkpoint_path}")

    if args.train_path:
        
        ##### TRAIN and VALIDATION PHASE #####

        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(12) 
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

        num_epochs = config[curr_dataset]["epochs"]
        best_val_accuracy = 0.0   

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        for epoch in range(num_epochs):
            if config[curr_dataset]["loss"] == 3: 
                train_loss,train_acc = train_co_teaching(train_loader,model1,model2,
                    optimizer1,optimizer2,criterion,device,
                    save_checkpoints=(epoch + 1 in checkpoint_intervals),
                    checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                    current_epoch=epoch)
            else:
                train_loss, train_acc = train(train_loader,model1,
                optimizer1,criterion,device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch)

            val_loss, val_acc, val_f1 = evaluate(val_loader, model1, device, calculate_accuracy=True)

        
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model1.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")

        # Plot
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

        import gc
        del train_dataset
        del train_loader
        del full_dataset
        del val_dataset
        del val_loader
        gc.collect()

    ##### TESTING PHASE #####

    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Generate predictions for the test set using the best model
    model1.load_state_dict(torch.load(checkpoint_path, map_location=device))
    predictions = evaluate(test_loader, model1, device, calculate_accuracy=False)
    save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    args = parser.parse_args()
    main(args)
