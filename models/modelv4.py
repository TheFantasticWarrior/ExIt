import torch
import torch.nn as nn
import torch.nn.functional as F
import args
from models.mobilenet import MobileNetV4
#torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
#torch.jit.enable_onednn_fusion(True)
torch.set_float32_matmul_precision("medium")

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
        x=self.l(x.view(-1,1))
        x=x.view(batch_size,-1,self.d_model)
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
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3,1), stride=1)

        # F6: Fully connected layer
        self.fc1 = nn.Linear(64, 32)


    def forward(self, x):
        # C1 -> ReLU -> S2
        x = self.pool1(F.relu(self.conv1(x)))

        # C3 -> ReLU -> S4
        x = self.pool2(F.relu(self.conv2(x)))

        # C5 -> ReLU
        x = F.relu(self.conv3(x))

        # Flatten the output from the last conv layer
        x = x.view(-1, 64)

        # F6 -> ReLU
        x = F.relu(self.fc1(x))


        return x

class BoardModel(nn.Module):
    def __init__(self,piece_shapes):
        super(BoardModel, self).__init__()
        self.b_cnn = LeNet5().to(memory_format=torch.channels_last)
        self.fc = nn.Linear(8, 32)
        self.embedding = nn.Embedding(8,32)
        self.attn = nn.MultiheadAttention(32,4,batch_first=True)
        self.pos_encoder = PiecePosition(32)
        self.p=torch.cat((torch.as_tensor(piece_shapes),torch.zeros(1,4,4,4)),0).to(torch.device("cuda"))
    def forward(self, t):
        batch_size=t.size(0)
        board=torch.zeros((batch_size,3,21,10),device=torch.device("cuda")).to(memory_format=torch.channels_last)
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

        b = self.b_cnn(board).view(batch_size,-1,32)

        p = self.embedding(hold_next.clone())#.view(batch_size,7,32)
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

class Model(nn.Module):
    def __init__(self,shapes):
        super(Model, self).__init__()
        bm=BoardModel(shapes).to(torch.device("cuda"))
        at=TransformerEncoder(4, 2, 8, 32, 32).to(torch.device("cuda"))
        l=nn.Linear(224,96).to(torch.device("cuda"))
        cm=MobileNetV4().to(torch.device("cuda")).to(memory_format=torch.channels_last)
        self.boardm = bm #torch.jit.trace(bm,torch.zeros(1,232,dtype=torch.int,device=torch.device("cuda")))
        self.atk_trans = at #torch.jit.trace(at,torch.zeros(1,35,device=torch.device("cuda")))
        self.linear0=l #torch.jit.trace(l,torch.zeros(1,432,device=torch.device("cuda")))
        self.combined_model=torch.jit.trace(cm,(torch.zeros(1,3,22,12,device=torch.device("cuda")),torch.zeros(1,96,device=torch.device("cuda"))))


    def forward(self,x):

        sum_atk=x[...,464]
        atk=x[...,465:]
        atk = atk * torch.sign(sum_atk)[:,None]

        own = x[..., :232]
        opp = x[...,232:464]
        own[..., 0] = sum_atk
        opp[..., 0] = -sum_atk
        atk = torch.unsqueeze(atk.float(), -1)

        own_out,own_board = self.boardm(own)
        opp_out,opp_board = self.boardm(opp)
        atk_out = self.atk_trans(atk)
        x = torch.cat((atk_out, own_out, opp_out), dim=1)
        x = F.relu(self.linear0(x))
        x=self.combined_model(own_board,x)
        cat=torch.cat((atk_out, opp_out, own_out), dim=1)
        return x,atk,cat,opp_board

    #@torch.jit.export
    def cond_forward(self, x: torch.Tensor, atk: torch.Tensor, y: torch.Tensor, opp_board: torch.Tensor, p2: bool = False, loss: bool = False):
        if p2:
            atk_out2 = self.atk_trans(-atk)
            y = F.relu(self.linear0(y))
            y=self.combined_model(opp_board,y)

            a=torch.cat((x.unsqueeze(1), y.unsqueeze(1)),1)

            if loss:
                return -torch.log(a)
            return a
        if loss:
            return -torch.log(x)
        else:
            x_ac=torch.multinomial(x, num_samples=1, replacement=True).squeeze()
            return x_ac

class Model2(nn.Module):
    def __init__(self,shapes,x):
        super(Model2, self).__init__()
        self.base=Model(shapes)
        self.traced_base=torch.jit.trace(self.base,x)
    def forward(self, x, p2: bool = False, loss: bool = False):

        x,atk,y,opp_board=self.traced_base(x)
        x=self.base.cond_forward(x,atk,y,opp_board,p2,loss)
        return x
class ModelWrapper:
    def __init__(self,shapes,x,lr=1e-4,gpu=True):
        self.model=Model2(shapes,x)
        if gpu:
            self.model.to(torch.device("cuda"))
        self.scripted_model = torch.jit.script(self.model)
        self.params=list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(self.params, 1e-4,weight_decay=0.01)
        self.lr=torch.optim.lr_scheduler.CyclicLR(self.optimizer,5e-5,max_lr=5e-4)

    def __call__(self,*args,**kwargs):
        return self.scripted_model(*args,**kwargs)
    def backward(self, x1, y1,loss_only:bool=False):
        cross1=x1*y1 #b,2,10
        kl1=torch.mean(torch.sum(y1*torch.log(y1)+cross1,-1))
        if not torch.isfinite(kl1):
            breakpoint()

        if not loss_only:
            loss = torch.mean(torch.sum(cross1,-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params,1)
            self.optimizer.step()
            self.lr.step()
            self.optimizer.zero_grad(set_to_none=True)

        return (kl1).item()
    def save(self,path="save_model"):
        torch.save({"model":self.model.base.state_dict(),
            "optim":self.optimizer.state_dict()},path)
    def load(self,path="save_model"):
        checkpoint = torch.load(path)
        i=0
        try:
            self.optimizer.load_state_dict(checkpoint['optim'])
        except:
            i=1
            print("no optimizer state found")
        try:
            self.model.base.load_state_dict(checkpoint['model'])
        except:
            print("no model state found")
            if i:
                raise Exception("Load model failed")

