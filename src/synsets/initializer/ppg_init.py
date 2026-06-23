from ppg.core.PPGB import PM
import torch
from my_utils.device import DeviceSingleton
from config import DistillCfg
from PIL import Image
import numpy as np
import torch.nn.functional as F

class PPGInitializer() :

    def __init__(self, cfg: DistillCfg): 
        self.cfg = cfg
        self.physical_model = self.load_physical_model(self.cfg.ppg_input_channels, self.cfg.ppg_checkpoint_path)        


    def load_physical_model(self, input_channels, checkpoint_path):
        model = PM(input_channels=input_channels)
        ckpt = torch.load(checkpoint_path, map_location=DeviceSingleton.get(), weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(DeviceSingleton.get())  
        return model

    def load_img(self, path):
        """PNG -> tensor [1,3,H,W] dans [-1,1]."""
        pil = Image.open(path).convert('RGB').resize((256, 256), Image.BICUBIC)
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float().to(DeviceSingleton.get())
    
    
    def run_physical_model(self, input_tensor):
        """Run the physical model on the input tensor."""
        with torch.no_grad():
            p_a, p_t = self.physical_model(input_tensor)

        syn_T_init = F.interpolate(p_t, (self.cfg.syn_res, self.cfg.syn_res),
                               mode='bilinear', align_corners=False)
        
        B01 = ((p_a + 1) / 2).mean(dim=(2, 3), keepdim=True).clamp(1e-4, 1 - 1e-4)
        syn_B_init = torch.log(B01 / (1 - B01))

        return syn_T_init, syn_B_init