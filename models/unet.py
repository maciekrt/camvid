
import torch
import torch.nn as nn
import timm
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet
from torch.nn import functional as F
import pytorch_lightning as pl


def get_dims(model, layer_names, size):
    save_hooks = {}
    for layer_name in layer_names:
        layer = dict(model.named_children())[layer_name]
        hook = SaveFeaturesHook(store_input=True, store_output=True)
        hook_handle = layer.register_forward_hook(hook)
        save_hooks[layer_name] = {"hook": hook, "hook_handle": hook_handle}
    dummy = torch.randn(1,3,*size).to("cpu")
    model.to("cpu")
    model(dummy)
    dims = {}
    for layer_name in save_hooks:
        input_shape = save_hooks[layer_name]["hook"].input.shape[1:]
        output_shape = save_hooks[layer_name]["hook"].output.shape[1:]
        dims[layer_name] = [input_shape, output_shape]
        save_hooks[layer_name]["hook_handle"].remove()
    return dims


class SaveFeaturesHook:
    
    def __init__(self, store_input=False, store_output=True, name="save_output"):
        self.store_input = store_input
        self.store_output = store_output
        self.input = None
        self.output = None
        self.name = name
        
    def __call__(self, module, input, output):
        if self.store_output:
            self.output = output
        if self.store_input:
            self.input = input[0]

            
class UpsampleUNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, output_size, hook_stored=None, activation="relu"):
        super(UpsampleUNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.hook_stored = hook_stored
        self.conv_transpose1 = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1)
        )
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "mish":
            self.activation = nn.Mish()
        else:
            raise ValueError(f"Activation function {activation} is not supported")
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.res_block1 = resnet.BasicBlock(self.out_channels, self.out_channels)
        
    def forward(self, x):
        if self.hook_stored is not None:
            skip_connection = self.hook_stored.output
            x = torch.cat([x, skip_connection], axis=1)
        x = self.conv_transpose1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.res_block1(x)
        x = F.interpolate(x, size=self.output_size, mode="nearest")
        return x
    

class ResizeToSize(nn.Module):
    
    def __init__(self, size):
        super(ResizeToSize, self).__init__()
        self.size = size
        
    def forward(self, x):
        return F.interpolate(x, size=self.size, mode="nearest")
    

class UNet(nn.Module):
    
    def __init__(self, encoder, size, layer_name="layer4", out_channels=2):
        super(UNet, self).__init__()
        self.encoder = encoder
        self.size = size
        self.out_channels=out_channels
        self.layer_name = layer_name
        self.save_hooks = {}
        layers = list(self.encoder.named_children())[4:-2]
        names = [name for name, _ in layers]
        self.names = names
        dims = get_dims(self.encoder, names, size)
        self.dims = dims
        
        modules = []
        last = 0
        for name, layer in reversed(layers):
            hook = SaveFeaturesHook(name=f"save_hook_{name}",store_input=(name == self.names[0]))
            self.save_hooks[name] = {
                "hook": hook,
                "hook_handle": layer.register_forward_hook(hook)
            }
            layer_input_channels = dims[name][0][0]
            layer_output_channels = dims[name][1][0]
            input_size = dims[name][0][-2:]
            output_size = dims[name][1][-2:]
            upsample = UpsampleUNetBlock(
                in_channels=layer_output_channels+last,
                out_channels=layer_input_channels,
                output_size=input_size,
                hook_stored=None if last == 0 else self.save_hooks[name]["hook"],
            )
            last = layer_input_channels
            modules.append(upsample)
        last //= 2 # Adjust to Pixel Shuffle
        self.upsample_model = nn.Sequential(*modules)
        self.head = nn.Sequential(*[
            nn.PixelShuffle(2),
            resnet.BasicBlock(last,last),
            nn.ConvTranspose2d(last,last,kernel_size=(3,3),stride=2, padding=1),
            nn.BatchNorm2d(last),
            ResizeToSize(size),
            nn.Conv2d(last, self.out_channels, kernel_size=(1,1), stride=1)
        ])
    
    def forward(self, x):
        self.encoder(x)
        x = self.save_hooks[self.names[-1]]["hook"].output
        skip_zero = self.save_hooks[self.names[0]]["hook"].input
        x = torch.cat([self.upsample_model(x), skip_zero], axis=1)
        return self.head(x)
