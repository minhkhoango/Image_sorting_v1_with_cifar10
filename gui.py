import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torch.nn as nn

# Load the trained model

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
model = torch.load(
    "trained_pikachu.pth", weights_only=False, 
    map_location=torch.device('cpu'))

model.eval()

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',  
           'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    image = preprocess_image(file_path)
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    result_label.config(text=f"Prediction: {classes[predicted_class]}", fg="white", bg="#2c3e50")
    
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Create GUI window
root = tk.Tk()
root.title("CIFAR-10 Image Classifier")
root.geometry("500x600")
root.configure(bg="#34495e")

frame = Frame(root, bg="#2c3e50", bd=5)
frame.pack(pady=20, padx=20, fill="both", expand=True)

btn_upload = Button(frame, text="Upload Image", command=classify_image, font=("Arial", 14), bg="#1abc9c", fg="white", padx=10, pady=5)
btn_upload.pack(pady=10)

image_label = Label(frame, bg="#2c3e50")
image_label.pack()

result_label = Label(frame, text="Prediction: ", font=("Arial", 16), fg="white", bg="#2c3e50")
result_label.pack(pady=10)

root.mainloop()

