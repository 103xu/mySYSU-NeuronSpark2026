import torch
import pandas
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from data import mydataset
test_dataset=mydataset(csv_path="test.csv",image_dir="test_images",has_label=False)
test_loader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=False)
model=models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features,10)
model.load_state_dict(torch.load("model.pth"))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
model.eval()
results=[]
with torch.no_grad():
    for images, labels in test_loader:
        images=images.to(device)
        outputs=model(images)
        predicted = torch.argmax(outputs, 1)
        for filename, predicted in zip(labels, predicted):
            print(filename,predicted)
            label_name= test_dataset.idx_to_label[predicted.item()]
            results.append([filename,label_name])
df = pandas.DataFrame(results,columns=["id", "label"])
df.to_csv("results.csv",index=False,encoding="utf-8-sig")
print("预测完成")
