import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import args


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
        self.fc = nn.Linear(3*d_model, output_dim)

    def forward(self, x):
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)
        x_mean = x.mean(dim=1)
        x_sum = x.sum(dim=1)
        x_max, _ = x.max(dim=1)
        x = torch.cat((x_mean, x_sum, x_max), 1)
        x = F.relu(self.fc(x))
        return x

class PiecePosition(nn.Module):
    def __init__(self, d_model, max_length=7):
        super(PiecePosition, self).__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.register_buffer('pe', self._generate_pe())

    def _generate_pe(self):
        pe = torch.zeros(self.max_length, self.d_model)
        position = torch.tensor([0,0,1,2,3,4,5])[:,None]
        div_term = torch.exp(torch.arange(0, self.d_model,2).float() * -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # C1: Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)

        # S2: Subsampling (Pooling) layer
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C3: Convolutional layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # S4: Subsampling (Pooling) layer
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C5: Fully connected convolutional layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(3,1), stride=1)

        # F6: Fully connected layer
        self.fc1 = nn.Linear(120, 84)


    def forward(self, x):
        # C1 -> ReLU -> S2
        x = self.pool1(F.relu(self.conv1(x)))

        # C3 -> ReLU -> S4
        x = self.pool2(F.relu(self.conv2(x)))

        # C5 -> ReLU
        x = F.relu(self.conv3(x))

        # Flatten the output from the last conv layer
        x = x.view(-1, 120)

        # F6 -> ReLU
        x = F.relu(self.fc1(x))


        return x

class BoardModel(nn.Module):
    def __init__(self,piece_shapes):
        super(BoardModel, self).__init__()
        self.b_cnn = LeNet5()
        self.fc = nn.Linear(8, 16)
        self.pfc = nn.Linear(1, 84)
        self.attn = nn.MultiheadAttention(84,14,batch_first=True)
        self.pos_encoder = PiecePosition(84)
        self.p=torch.cat((torch.tensor(piece_shapes),torch.zeros(1,4,4,4)),0).to(torch.device("cuda"))
    def forward(self, t):
        batch_size=t.size(0)
        board=torch.zeros((batch_size,3,21,10),device=torch.device("cuda"))
        board[:,0] = t[:, 22:232].view(batch_size, 21, 10)
        #draw ghost and shadow
        if torch.any(t[:,8]>6) or torch.any(t[:,8]<0):
            print(t[:,8])
            breakpoint()
        if torch.any(t[:,4]>3) or torch.any(t[:,4]<0):
            print(t[:,4])
            breakpoint()
        pieces=self.p[t[:,8].int(),t[:,4].int()]
        a,y,x=torch.where(pieces)
        x+=torch.repeat_interleave(t[:,1].int(),4)-2
        y+=torch.repeat_interleave(t[:,2].int(),4)
        ny=y+torch.repeat_interleave(t[:,3].int(),4)

        mask=(y>=0)&(ny>=0)
        a=a[mask]
        y=y[mask]
        ny=ny[mask]
        x=x[mask]

        board[a,1,y,x]=1
        board[a,2,ny,x] = 1
        board=F.pad(board,(1, 1, 0, 1, 0, 0, 0, 0), value=1)
        t[:,9]=torch.where((t[:,9]!=-1) & (t[:,6]!=0),t[:,9],7)
        hold_next=t[:,8:15]

        b = self.b_cnn(board).view(batch_size,-1,84)

        p = self.pfc(hold_next.reshape(batch_size*7,1)).view(batch_size,7,84)
        p = self.pos_encoder(p)

        a,_ = self.attn(p,b,b,need_weights=False)
        x = F.relu(self.fc(t[:, :8]))

        x = torch.cat((x, a.mean(dim=1)), dim=1)
        return x


class Model(nn.Module):
    def __init__(self,shapes):
        super(Model, self).__init__()
        self.boardm = BoardModel(shapes)
        self.atk_trans = TransformerEncoder(4, 2, 4, 32, 84)
        self.linear0=nn.Linear(232,128)
        #self.linear1 = nn.Linear(512, 128)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        if args.gpu:
            self.to(torch.device("cuda"))
        self.optimizer = torch.optim.Adam(self.parameters(), 2.5e-4)
        # self.softmax = nn.Softmax()

    def forward(self, x, p2=False, loss=False,mask=None,fin=False):
        #x = np.array(x)
        x=np.atleast_2d(x)
        sum_atk=x[...,464]
        atk=x[...,465:]
        atk = atk * np.sign(sum_atk)[:,None]

        own = x[..., :232]
        opp = x[...,232:464]

        #ns = [torch.tensor([0]) if len(a) == 0 else torch.tensor(a)
        #      for a in atk]
        #atk = nn.utils.rnn.pad_sequence(ns if len(
        #    ns) else torch.repeat_interleave(torch.Tensor([0]), len(x)), True)

        own[..., 0] = sum_atk
        opp[..., 0] = -sum_atk
        atk = torch.unsqueeze(torch.tensor(atk, dtype=torch.float), -1)
        #print(torch.squeeze(atk))
        own = torch.tensor(own, dtype=torch.float)
        opp = torch.tensor(opp, dtype=torch.float)
        if args.gpu:
            opp = opp.to(torch.device("cuda"))
            own = own.to(torch.device("cuda"))
            atk = atk.to(torch.device("cuda"))

        atk_out = self.atk_trans(atk)
        own_out = self.boardm(own)
        opp_out = self.boardm(opp)
        x = torch.cat((atk_out, own_out, opp_out), dim=1)
        x = self.linear0(x)
        x = F.softmax(self.linear2(x), dim=-1)

        if p2:
            atk_out2 = self.atk_trans(-atk)
            y = torch.cat((atk_out, own_out, opp_out), dim=1)
            y = self.linear0(y)
            y = F.softmax(self.linear2(y), dim=-1)
            if fin:
                return x,y
            elif loss:
                neglogpy = -torch.log(y)
                neglogpx = -torch.log(x)
                return neglogpx, neglogpy
            elif np.any(mask):
                masked_indices = self.apply_mask(torch.atleast_2d(x),torch.atleast_2d(y), torch.tensor(mask,dtype=torch.bool).to(torch.device("cuda")))
                return masked_indices

            x = torch.multinomial(x, num_samples=1, replacement=True).squeeze()
            y = torch.multinomial(y, num_samples=1, replacement=True).squeeze()
            if x.nelement() != 1:
                return torch.stack((x, y), 1)
            else:
                return torch.stack((x.unsqueeze(0), y.unsqueeze(0)), 1)

        if loss:
            return -torch.log(x)
        else:
            return torch.multinomial(x, num_samples=1, replacement=True).squeeze()

    def process_input(self, atk_out, own_out, opp_out):

        return x

    def apply_mask(self, arr1, arr2, mask):
        joint_probs = arr1.unsqueeze(2) * arr2.unsqueeze(1)

        joint_probs_masked = joint_probs * mask

        joint_probs_masked_flat = joint_probs_masked.view(joint_probs_masked.shape[0], -1)

        indices = torch.multinomial(joint_probs_masked_flat, 1)

        index1 = indices // arr1.shape[1]
        index2 = indices % arr1.shape[1]

        samples = torch.cat((index1,index2), dim=1)
        return samples

    def backward(self, x1, y1, x2, y2):
        y1, y2 = torch.tensor(y1), torch.tensor(y2)
        if args.gpu:
            y1, y2 = y1.to(torch.device("cuda")), y2.to(torch.device("cuda"))
        loss = (torch.nanmean(x1*y1)+torch.nanmean(x2*y2))/2
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
