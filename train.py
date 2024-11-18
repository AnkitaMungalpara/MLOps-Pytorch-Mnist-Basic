import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
import torch.nn.functional as F 
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_transform(train=True):
    if train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(10),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return transform

def show_augmented_samples():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(70),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('data', train=True, download=True, 
                           transform=transform)
    
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        img, label = dataset[i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    plt.close()

def train_model():
    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = get_transform(train=True)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=4, shuffle=True)
    
    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    
    # Training
    model.train()
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        
        progress_bar.set_postfix({'acc': f'{100. * correct / total:.2f}%'})
    
    accuracy = 100. * correct / total
    return accuracy

if __name__ == "__main__":
    show_augmented_samples()
    # accuracy = train_model()
    # print(f"Training Accuracy: {accuracy:.2f}%")
    
    # # Count parameters
    # model = MNISTModel()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")
