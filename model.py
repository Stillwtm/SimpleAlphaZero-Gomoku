import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
    
class Encoder(nn.Module):
    """Encode a game state.
    """
    def __init__(self, in_dim, num_layers):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.residual_layers = nn.Sequential(
            *[BasicBlock(256, 256) for _ in range(num_layers)]
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.residual_layers(out)
        
        return out

class PolicyNet(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        
        self.conv = nn.Conv2d(256, 2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2 * board_size**2, board_size**2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(x.shape[0], -1)

        out = self.fc(out)
        out = F.softmax(out, dim=1)

        return out

class ValueNet(nn.Module):
    def __init__(self, board_size):
        super().__init__()
        
        self.conv = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(board_size**2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(x.shape[0], -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.tanh(out)

        return out

class PolicyValueNet(nn.Module):
    def __init__(self, board_size, history, lr, num_layers=4, device='cuda'):
        super().__init__()

        self.device = device

        self.encoder = Encoder(in_dim=history*2+1, num_layers=num_layers).to(device)
        self.policy_head = PolicyNet(board_size).to(device)
        self.value_head = ValueNet(board_size).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.to(self.device)
        feat = self.encoder(x)
        action_probs = self.policy_head(feat)
        value = self.value_head(feat)

        return action_probs, value

    def update(self, states, mcts_probs, winners, device):
        """Perform a training step.

        Args:
            states (ndarray)
            mcts_probs (ndarray)
            winners (ndarray)
            device (str | torch.device): Working device.

        Returns:
            tuple: (loss, act_entropy)
        """
        states = torch.FloatTensor(states).to(device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(device)
        winners = torch.FloatTensor(winners).to(device)

        act_probs, value = self.forward(states)
        # Loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        value_loss = F.mse_loss(value.view(-1), winners)
        policy_loss = -torch.mean(torch.sum(mcts_probs * torch.log(act_probs), dim=1))
        loss = value_loss + policy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Calc policy entropy, for monitoring only
        act_entropy = -torch.mean(torch.sum(act_probs * torch.log(act_probs), 1))
        
        return loss.item(), act_entropy.item()

