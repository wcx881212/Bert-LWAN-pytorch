import torch
from torch.nn.init import xavier_uniform
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from neural_networks.custom_layers.dropout import SpatialDropout
from neural_networks.custom_layers.masking import Camouflage

class Model(nn.Module):
    def __init__(self, n_classes=4654, dropout_rate=0.5):
        super(Model, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = SpatialDropout(drop=dropout_rate)
        self.masking = Camouflage(mask_value=0)
        self.U = nn.Linear(768,n_classes)  # 输入 输出
        xavier_uniform(self.U.weight)
        self.final = nn.Linear(768,n_classes)
        xavier_uniform(self.final.weight)

    def forward(self,x_batch):
        bert_output = self.bert(x_batch)  # 32 512 768
        inner_out = self.dropout(bert_output[0])
        x = self.masking(inputs=[inner_out, bert_output[0]])  # 32 512 768
        alpha = F.softmax(torch.transpose(self.U(x),1,2), dim=-1)#32 4654 512
        m = alpha.matmul(x)#torch.Size([32, 4654, 768])
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)#torch.Size([32, 4654])
        y = torch.sigmoid(y)
        return y

