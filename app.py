import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Load model class
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(10 + 100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 28*28),
            torch.nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        out = self.fc(x)
        return out.view(-1, 1, 28, 28)

# Load trained model
device = 'cpu'
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator_mnist.pth', map_location=device))
generator.eval()

st.title("Handwritten Digit Generator")

digit = st.selectbox("Select a digit (0-9):", list(range(10)))

if st.button("Generate"):
    noise = torch.randn(5, 100)
    labels_onehot = F.one_hot(torch.tensor([digit]*5), num_classes=10).float()

    with torch.no_grad():
        fake_images = generator(noise, labels_onehot)

    grid = make_grid(fake_images, nrow=5, normalize=True)

    plt.figure(figsize=(10, 2))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu())
    st.pyplot(plt)
