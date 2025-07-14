import torch
import torch.nn as nn
import torch.optim as optim



class HeteroscedasticGaussianRegressor(nn.Module):
    """
    多维高斯异方差模型
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.linear_mean = nn.Linear(input_dim, 1, bias=True)  # μ(x) = xᵀβ
        # self.linear_mean = nn.Sequential(                    # μ(x) = xᵀβ
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim), 
        #     nn.ReLU(),
            
        #     nn.Linear(hidden_dim, 1)
        # )
        self.variance_net = nn.Sequential(                    # log σ²(x) = h(x)
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        mu = self.linear_mean(x).squeeze(-1)
        log_sigma2 = self.variance_net(x).squeeze(-1)
        sigma2 = torch.exp(log_sigma2)
        return mu, sigma2

    def nll_loss(self, x, y):
        mu, sigma2 = self.forward(x)
        return 0.5 * torch.mean(torch.log(sigma2) + (y - mu)**2 / sigma2)

    def mse_loss(self, x, y):
        mu = self.linear_mean(x).squeeze(-1)
        return torch.mean((y - mu) ** 2)
    
    
    
# def train_two_stage(model, x, y, stage1_epochs=3000, stage2_epochs=3000, lr=1e-3):
#     print("=== Stage 1: Train β using OLS (MSE loss) ===")
#     for p in model.linear_mean.parameters():
#         p.requires_grad = True
#     for p in model.variance_net.parameters():
#         p.requires_grad = False

#     optimizer = optim.Adam(model.linear_mean.parameters(), lr=lr)
#     for epoch in range(stage1_epochs):
#         optimizer.zero_grad()
#         loss = model.mse_loss(x, y)
#         loss.backward()
#         optimizer.step()
#         if epoch % 100 == 0:
#             print(f"[Stage 1][Epoch {epoch}] MSE: {loss.item():.4f}")

#     print("\n=== Stage 2: Fix β, Train h(x) for σ²(x) ===")
#     for p in model.linear_mean.parameters():
#         p.requires_grad = False
#     for p in model.variance_net.parameters():
#         p.requires_grad = True

#     optimizer = optim.Adam(model.variance_net.parameters(), lr=lr)
#     for epoch in range(stage2_epochs):
#         optimizer.zero_grad()
#         loss = model.nll_loss(x, y)
#         loss.backward()
#         optimizer.step()
#         if epoch % 200 == 0:
#             print(f"[Stage 2][Epoch {epoch}] NLL: {loss.item():.4f}")

def train_two_stage(model, x, y, stage1_epochs=5000, stage2_epochs=3000, lr=1e-3):
    optimizer_mean = optim.Adam(model.linear_mean.parameters(), lr=lr)
    optimizer_var = optim.Adam(model.variance_net.parameters(), lr=lr)

    for epoch in range(stage1_epochs):
        # Train mean
        for p in model.linear_mean.parameters():
            p.requires_grad = True
        for p in model.variance_net.parameters():
            p.requires_grad = False
        optimizer_mean.zero_grad()
        mse_loss = model.mse_loss(x, y)
        mse_loss.backward()
        optimizer_mean.step()

        # Train variance
        for p in model.linear_mean.parameters():
            p.requires_grad = False
        for p in model.variance_net.parameters():
            p.requires_grad = True
        optimizer_var.zero_grad()
        nll_loss = model.nll_loss(x, y)
        nll_loss.backward()
        optimizer_var.step()

        if epoch % 50 == 0:
            print(f"[Epoch {epoch}] MSE: {mse_loss.item():.4f}, NLL: {nll_loss.item():.4f}")