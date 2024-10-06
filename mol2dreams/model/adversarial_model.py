from torch import nn

class Discriminator(nn.Module):
    def __init__(self, embedding_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Embedding tensor of shape [batch_size, embedding_dim]

        Returns:
            output: Tensor of shape [batch_size], with values between 0 and 1
        """
        output = self.model(x).squeeze()
        return output