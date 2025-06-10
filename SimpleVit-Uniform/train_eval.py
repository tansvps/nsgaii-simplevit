import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class Train:        
    def traintest(self,model,train_loader,valid_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = 32
        criterion = nn.CrossEntropyLoss()
        lr = 3e-5
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.001)
        model.to(device)
        for epoch in range(epochs): 
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
        
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            # print(f'Epoch {epoch+1}/{epochs}   Train Loss:{train_loss:.3f}   Train Accuracy:{train_accuracy:.2f}%    ')
            
            # Evaluate on test data
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in valid_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device) 
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
            test_loss /= len(valid_loader)
            test_accuracy = 100 * correct / total
            # print(f'Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%')

            # print(f'Epoch {epoch+1}/{epochs}   Train Loss:{train_loss:.3f}   Test Loss: {test_loss:.3f}    Train Accuracy:{train_accuracy:.2f}%    Test Accuracy: {test_accuracy:.2f}%')
        # print("test_accuracy",test_accuracy)
        print('Finished Training!')
        return test_accuracy