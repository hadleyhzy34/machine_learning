"""
Simple GAN Demo with PyTorch
============================

This demo shows how Generative Adversarial Networks (GANs) work.

GANs consist of two neural networks competing against each other:
1. Generator (G): Creates fake images from random noise
2. Discriminator (D): Distinguishes real images from fake ones

They play a minimax game:
- D tries to maximize its ability to distinguish real from fake
- G tries to fool D by generating realistic images

Training objective:
- D loss: maximize log(D(x)) + log(1 - D(G(z)))
- G loss: minimize log(1 - D(G(z))) or maximize log(D(G(z)))
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================
# 1. Define the Generator Network
# ============================================
class Generator(nn.Module):
    """
    Generator: Takes random noise and generates fake images.

    Architecture: noise -> FC -> ReLU -> FC -> Tanh -> image
    """

    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=784):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z):
        # z: (batch_size, latent_dim)
        img = self.model(z)
        return img.view(-1, 1, 28, 28)  # Reshape to image (1, 28, 28)


# ============================================
# 2. Define the Discriminator Network
# ============================================
class Discriminator(nn.Module):
    """
    Discriminator: Classifies images as real or fake.

    Architecture: image -> FC -> LeakyReLU -> FC -> Sigmoid -> probability
    """

    def __init__(self, input_dim=784, hidden_dim=256):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # LeakyReLU works better for GANs
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output probability [0, 1]
        )

    def forward(self, img):
        # img: (batch_size, 1, 28, 28)
        img_flat = img.view(-1, 784)  # Flatten image
        validity = self.model(img_flat)
        return validity


# ============================================
# 3. Hyperparameters and Data Loading
# ============================================
latent_dim = 100  # Size of random noise vector
hidden_dim = 256  # Hidden layer size
batch_size = 64  # Batch size
lr = 0.0002  # Learning rate
num_epochs = 50  # Number of training epochs

# Transform for MNIST: normalize to [-1, 1] to match Tanh output
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
    ]
)

# Load MNIST dataset
dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset size: {len(dataset)} images")


# ============================================
# 4. Initialize Models, Optimizers, Loss
# ============================================
# Create models
generator = Generator(latent_dim, hidden_dim).to(device)
discriminator = Discriminator(784, hidden_dim).to(device)

# Optimizers (Adam works well for GANs)
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Binary Cross Entropy Loss
adversarial_loss = nn.BCELoss()

print("\nGenerator Architecture:")
print(generator)
print("\nDiscriminator Architecture:")
print(discriminator)


# ============================================
# 5. Training Loop
# ============================================
def train_gan():
    """Train the GAN and return training history."""

    # For tracking progress
    G_losses = []
    D_losses = []
    generated_images = []

    # Fixed noise for consistent visualization
    fixed_noise = torch.randn(16, latent_dim, device=device)

    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)

    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0

        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size_curr = real_imgs.shape[0]
            real_imgs = real_imgs.to(device)

            # =====================================
            # Train Discriminator
            # =====================================
            # Goal: D(x) -> 1, D(G(z)) -> 0

            d_optimizer.zero_grad()

            # Labels for real and fake
            real_labels = torch.ones(batch_size_curr, 1, device=device)
            fake_labels = torch.zeros(batch_size_curr, 1, device=device)

            # Loss on real images
            real_output = discriminator(real_imgs)
            d_loss_real = adversarial_loss(real_output, real_labels)

            # Generate fake images
            noise = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_imgs = generator(noise)

            # Loss on fake images (detach to not update G here)
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = adversarial_loss(fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # =====================================
            # Train Generator
            # =====================================
            # Goal: D(G(z)) -> 1 (fool the discriminator)

            g_optimizer.zero_grad()

            # Generate new fake images
            noise = torch.randn(batch_size_curr, latent_dim, device=device)
            fake_imgs = generator(noise)

            # We want discriminator to classify fakes as real (label=1)
            output = discriminator(fake_imgs)
            g_loss = adversarial_loss(output, real_labels)

            g_loss.backward()
            g_optimizer.step()

            # Track losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        # Average losses for this epoch
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        D_losses.append(avg_d_loss)
        G_losses.append(avg_g_loss)

        # Generate images with fixed noise for visualization
        with torch.no_grad():
            fake = generator(fixed_noise).cpu()
            generated_images.append(fake)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f}"
            )

    return G_losses, D_losses, generated_images


# Run training
G_losses, D_losses, generated_images = train_gan()


# ============================================
# 6. Visualization
# ============================================


def show_generated_images(images, title="Generated Images"):
    """Display a grid of generated images."""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_losses(G_losses, D_losses):
    """Plot training losses."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(G_losses, label="Generator Loss")
    ax.plot(D_losses, label="Discriminator Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Progress")
    ax.legend()
    ax.grid(True)
    return fig


# Plot training losses
fig1 = plot_losses(G_losses, D_losses)
plt.savefig("gan_training_losses.png", dpi=150, bbox_inches="tight")
print("\nSaved: gan_training_losses.png")

# Show initial vs final generated images
fig2 = show_generated_images(generated_images[0], "Generated Images (Epoch 1)")
plt.savefig("gan_generated_epoch1.png", dpi=150, bbox_inches="tight")
print("Saved: gan_generated_epoch1.png")

fig3 = show_generated_images(
    generated_images[-1], f"Generated Images (Epoch {num_epochs})"
)
plt.savefig("gan_generated_final.png", dpi=150, bbox_inches="tight")
print("Saved: gan_generated_final.png")

# Create a progression GIF-like visualization
fig4, axes = plt.subplots(2, 5, figsize=(15, 6))
epochs_to_show = (
    [0, 4, 9, 19, 29, 39, 44, 49]
    if num_epochs >= 50
    else list(range(0, num_epochs, max(1, num_epochs // 8)))
)
epochs_to_show = epochs_to_show[:10]  # Max 10 images

for idx, epoch in enumerate(epochs_to_show):
    row, col = idx // 5, idx % 5
    if row < 2 and col < 5:
        img = generated_images[epoch][0].squeeze().numpy()
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(f"Epoch {epoch + 1}")
        axes[row, col].axis("off")

plt.suptitle("GAN Training Progression")
plt.tight_layout()
plt.savefig("gan_progression.png", dpi=150, bbox_inches="tight")
print("Saved: gan_progression.png")

plt.show()


# ============================================
# 7. Interactive Demo: Generate New Images
# ============================================
print("\n" + "=" * 50)
print("Generating new images with trained generator...")
print("=" * 50)

with torch.no_grad():
    # Generate some new random images
    noise = torch.randn(16, latent_dim, device=device)
    generated = generator(noise).cpu()

    fig5 = show_generated_images(generated, "New Generated Images")
    plt.savefig("gan_new_generated.png", dpi=150, bbox_inches="tight")
    print("Saved: gan_new_generated.png")

print("\nTraining complete!")
print("\nKey takeaways:")
print("1. The generator learns to create realistic images from random noise")
print("2. The discriminator learns to distinguish real from fake images")
print("3. Through adversarial training, both networks improve together")
print("4. Eventually, the generator creates images that fool the discriminator")

