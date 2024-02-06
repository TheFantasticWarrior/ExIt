import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        pe = torch.zeros(x.size(0), x.size(1), self.d_model).to(x.device)
        position = torch.arange(0, x.size(1), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, self.d_model, 2).float() * -(np.log(10000.0) / self.d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return x + pe


class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, num_layers, d_model, output_dim, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True), num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


class ImpalaCNN(nn.Module):
    def __init__(self, in_c):
        super(ImpalaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 16, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.res2 = ResidualBlock(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.res3 = ResidualBlock(32)
        self.fc = nn.Linear(576, 256)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)
        x = F.relu(x)
        x = self.fc(x.view(x.size(0), -1))
        x = F.relu(x)
        return x


class SmallCNN(nn.Module):
    def __init__(self, in_c):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0)
        self.fc = nn.Linear(288, 256)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.fc(x.view(x.size(0), -1))

        return x


class BoardModel(nn.Module):
    def __init__(self):
        super(BoardModel, self).__init__()
        self.b_cnn = ImpalaCNN(3)
        self.p_cnn = SmallCNN(6)
        self.fc = nn.Linear(12, 128)

    def forward(self, x: torch.Tensor):
        board = F.pad(x[..., 12:642].reshape(-1, 3, 10, 21),
                      (0, 1, 1, 1, 0, 0, 0, 0), value=1)
        hold_next = x[..., 642:].reshape(-1, 6, 4, 4)
        b = self.b_cnn(board)
        p = self.p_cnn(hold_next)
        x = F.relu(self.fc(x[..., :12]))
        x = torch.cat((x, b, p), dim=1)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.boardm = BoardModel()
        self.atk_trans = TransformerEncoder(4, 2, 4, 128, 128)
        self.trans = TransformerEncoder(8, 8, 1408, 512, 1024)
        self.linear1 = nn.Linear(512, 128)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.optimizer = torch.optim.Adam(self.parameters(), 3e-4)
        # self.softmax = nn.Softmax()

    def forward(self, x, y, p2=False, loss=False):
        if type(y[0]) == bool:
            to_self, atk = y
            to_self = np.array(to_self).reshape(1, -1)
            x = np.array(x).reshape(1, -1)
            if len(atk):
                atk = atk.reshape(1, -1)
            else:
                atk = np.zeros_like(to_self, dtype=np.float32)
        else:
            try:
                to_self, atk = zip(*y)
            except:
                breakpoint()
            x = np.array(x)

        atk = atk if to_self else -atk

        own = x[..., :738]
        opp = x[..., 738:1476]

        ns = [torch.tensor([0]) if len(a) == 0 else torch.tensor(a)
              for a in atk]
        atk = nn.utils.rnn.pad_sequence(ns if len(
            ns) else torch.repeat_interleave(torch.Tensor([0]), len(x)), True)

        try:
            own[..., 0] = torch.sum(atk, -1)
        except:
            breakpoint()
        opp[..., 0] = -own[..., 0]
        atk = torch.unsqueeze(atk, -1)

        own = torch.tensor(own, dtype=torch.float).to(torch.device("cuda"))
        opp = torch.tensor(opp, dtype=torch.float).to(torch.device("cuda"))
        atk = atk.to(torch.device("cuda"))

        atk_out = self.atk_trans(atk)
        own_out = self.boardm(own)
        opp_out = self.boardm(opp)

        x = torch.unsqueeze(torch.cat((atk_out, own_out, opp_out), dim=1), 1)
        x = torch.squeeze(self.trans(x))
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        if loss:
            neglogpx = -torch.log(F.softmax(x, dim=-1))
        else:
            x = torch.max(F.gumbel_softmax(x), dim=-1)[1]

        if p2:
            atk2 = -atk
            atk_out2 = self.atk_trans(atk2)
            y = torch.unsqueeze(
                torch.cat((atk_out2, opp_out, own_out), dim=1), 1)
            y = torch.squeeze(self.trans(y))
            y = self.linear1(y)
            y = self.activation(y)
            y = self.linear2(y)
            neglogpy = -torch.log(F.softmax(y, dim=-1))

            if loss:
                return neglogpx, neglogpy
            y = torch.max(F.gumbel_softmax(y), dim=-1)[1]
            if x.nelement() != 1:
                return torch.stack((x, y), 1)
            else:
                x.unsqueeze_(0)
                y.unsqueeze_(0)
                return torch.stack((x, y), 1)
        if loss:
            return neglogpx
        return x

    def backward(self, x1, y1, x2, y2):
        y1, y2 = map(torch.tensor, (y1, y2))
        loss = (torch.nanmean(x1*y1)+torch.nanmean(x2*y2))/2
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
