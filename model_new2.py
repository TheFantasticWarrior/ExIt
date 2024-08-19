import torch
import torch.nn as nn
import torch.nn.functional as F
import args
torch.autograd.set_detect_anomaly(True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        pe = torch.zeros(x.size(0), x.size(1), self.d_model).to(x.device)
        position = torch.arange(0, x.size(1), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, self.d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return x + pe


class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, num_layers, d_model, output_dim, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.d_model=d_model
        self.l=nn.Linear(1,d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True), num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc = nn.Linear(3*d_model, output_dim)

    def forward(self, x):
        batch_size=x.size(0)
        x=self.l(x.view(-1,1)).view(batch_size,-1,self.d_model)
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
        div_term = torch.exp(torch.arange(0, self.d_model,2).float() * -(torch.log(torch.tensor(10000.0)) / self.d_model))
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
        self.fc = nn.Linear(8, 32)
        self.embedding = nn.Embedding(8,84)
        self.attn = nn.MultiheadAttention(84,14,batch_first=True)
        self.pos_encoder = PiecePosition(84)
        self.p=torch.cat((torch.as_tensor(piece_shapes),torch.zeros(1,4,4,4)),0).to(torch.device("cuda"))
    def forward(self, t):
        batch_size=t.size(0)
        board=torch.zeros((batch_size,3,21,10),device=torch.device("cuda"))
        board[:,0] = t[:, 22:232].float().view(batch_size, 21, 10)
        #draw ghost and shadow
        """if torch.any(t[:,8]>6) or torch.any(t[:,8]<0):
            print(t[:,8])
            breakpoint()
        if torch.any(t[:,4]>3) or torch.any(t[:,4]<0):
            print(t[:,4])
            breakpoint()"""
        pieces=self.p[t[:,8],t[:,4]]
        a,y,x=torch.where(pieces)
        x=x+torch.repeat_interleave(t[:,1],4)-2
        y=y+torch.repeat_interleave(t[:,2],4)
        ny=y+torch.repeat_interleave(t[:,3],4)

        mask=(y>=0)&(ny>=0)
        a=a[mask]
        y=y[mask]
        ny=ny[mask]
        x=x[mask]
        board[a,1,y,x]=1
        board[a,2,ny,x] = 1
        board=F.pad(board,(1, 1, 0, 1, 0, 0, 0, 0), value=1.)
        t[:,9]=torch.where((t[:,9]!=-1) & (t[:,6]!=0),t[:,9],7)
        hold_next=t[:,8:15]

        b = self.b_cnn(board).view(batch_size,-1,84)

        p = self.embedding(hold_next.clone())#.view(batch_size,7,84)
        p = self.pos_encoder(p)

        a,_ = self.attn(p,b,b)
        x = F.relu(self.fc(t[:, :8].float()))

        x = torch.cat((x, a.mean(dim=1),b.squeeze(1)), dim=1)
        return x,board

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
        self.res12 = ResidualBlock(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.res2 = ResidualBlock(32)
        self.res22 = ResidualBlock(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.res3 = ResidualBlock(32)
        self.res32 = ResidualBlock(32)
        self.fc = nn.Linear(192, 256)
        self.fc2 = nn.Linear(256,10)

    def forward(self, x,c):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res1(x)
        x = self.res12(x)


        x = F.relu(self.conv2(x))
        x=x+c.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,x.size(2),x.size(3))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res2(x)
        x = self.res22(x)


        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res3(x)
        x = self.res32(x)

        x = self.fc(F.relu(x.view(x.size(0), -1)))
        x=F.softmax(self.fc2(F.relu(x)),-1)
        return x

class Model(nn.Module):
    def __init__(self,shapes):
        super(Model, self).__init__()
        bm=BoardModel(shapes).to(torch.device("cuda"))
        at=TransformerEncoder(4, 2, 8, 32, 84).to(torch.device("cuda"))
        l=nn.Linear(432,32).to(torch.device("cuda"))
        cm=ImpalaCNN(3).to(torch.device("cuda"))
        self.boardm = torch.jit.trace(bm,torch.zeros(1,232,dtype=torch.int,device=torch.device("cuda")))
        self.atk_trans = torch.jit.trace(at,torch.zeros(1,35,device=torch.device("cuda")),check_trace=False)
        self.linear0=torch.jit.trace(l,torch.zeros(1,432,device=torch.device("cuda")))
        self.combined_model=torch.jit.trace(cm,(torch.zeros(1,3,22,12,device=torch.device("cuda")),torch.zeros(1,32,device=torch.device("cuda"))))
        if args.gpu:
            self.to(torch.device("cuda"))
        self.params=self.parameters()
        self.optimizer = torch.optim.AdamW(self.parameters(), 0.0025)

    def forward(self, x, p2: bool = False, loss: bool = False, mask: torch.Tensor = torch.zeros(2,10), fin: bool = False):

        x,atk,y,opp_board=self.base_forward(x)
        x=self.cond_forward(x,atk,y,opp_board,p2,loss,mask,fin)
        return x

    def base_forward(self,x):

        sum_atk=x[...,464]
        atk=x[...,465:]
        atk = atk * torch.sign(sum_atk)[:,None]

        own = x[..., :232]
        opp = x[...,232:464]
        own[..., 0] = sum_atk
        opp[..., 0] = -sum_atk
        atk = torch.unsqueeze(atk.float(), -1)

        atk_out = self.atk_trans(atk)
        own_out,own_board = self.boardm(own)
        opp_out,opp_board = self.boardm(opp)
        x = torch.cat((atk_out, own_out, opp_out), dim=1)
        x = F.relu(self.linear0(x))
        x=self.combined_model(own_board,x)
        cat=torch.cat((atk_out, opp_out, own_out), dim=1)
        return x,atk,cat,opp_board

    #@torch.jit.export
    def cond_forward(self, x: torch.Tensor, atk: torch.Tensor, y: torch.Tensor, opp_board: torch.Tensor, p2: bool = False, loss: bool = False, mask: torch.Tensor = torch.zeros(2,10), fin: bool = False):
        if p2:
            atk_out2 = self.atk_trans(-atk)
            y = F.relu(self.linear0(y))
            y=self.combined_model(opp_board,y)
            if fin:
                return x,y
            elif loss:
                neglogpy = -torch.log(y)
                neglogpx = -torch.log(x)
                return neglogpx, neglogpy
            elif torch.any(mask):
                masked_indices = self.apply_mask(torch.atleast_2d(x),torch.atleast_2d(y), torch.tensor(mask,dtype=torch.bool).to(torch.device("cuda")))
                return masked_indices

            x = torch.multinomial(x, num_samples=1, replacement=True).squeeze()
            y = torch.multinomial(y, num_samples=1, replacement=True).squeeze()
            return x,y
            if x.nelement() != 1:
                return x,y
            else:
                return x.unsqueeze(0), y.unsqueeze(0)

        if loss:
            return -torch.log(x),torch.empty_like(x)
        else:
            x_ac=torch.multinomial(x, num_samples=1, replacement=True).squeeze()
            return x_ac,torch.empty_like(x_ac)


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
        return samples[0],samples[1]

