import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from data import mydataset
dataset=mydataset(csv_path="labels.csv",image_dir="images",has_label=True)
loader = DataLoader(dataset=dataset,batch_size=16,shuffle=True)
model=models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features,10)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, "f"Loss={running_loss:.4f}")
    print(f"Epoch {epoch + 1}, "f"Loss={running_loss:.4f}")
torch.save(model.state_dict(), "model.pth")
print("Finished Training")