import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class LitMNISTModel(pl.LightningModule):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_size, hidden_size)
        self.layer_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Flatten the input tensor
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1)

        # Forward pass
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    # Data
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=64)
    val_loader = DataLoader(mnist_val, batch_size=64)
    test_loader = DataLoader(mnist_test, batch_size=64)

    # Model
    model = LitMNISTModel()

    # Training
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_loader, val_loader)

    # Testing
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
