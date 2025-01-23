import torch
from typing import Tuple, Dict, Optional, Any
from omegaconf import DictConfig


class GWGenerator():
    """
    GWGenerator:
        GGWGenerator for FL clients, which computes/generates the local embeddings using the given batch
    """  
    def __init__(
        self,
        model: torch.nn.Module=None,
        generator_configs: DictConfig = DictConfig({}),
        logger: Optional[Any]=None,
        **kwargs
    ):
        
        self.model = model
        self.generator_configs = generator_configs
        self.logger = logger
        self.__dict__.update(kwargs)

        if not hasattr(self.generator_configs, "device"):
            self.generator_configs.device = "cpu"

        self.model.to(self.generator_configs.device)

                                

    def get_parameters(self) -> Dict:
        if self.generator_configs.do_inference==False:
            return
            # return {
            #     'train_embedding': self.train_embedding.detach().clone().cpu(),
            #     'val_embedding': self.val_embedding.cpu(),
            # }
        else:
            return {
                'inference_embedding': self.inference_embedding.cpu()
            }


    def compute_embeddings(self, inference_data):
        # 1 batch of windows
        with torch.no_grad():
            self.model.eval()
            # Move the data to appropriate device
            inference_tensor = inference_data.to(self.generator_configs.device)
            self.inference_embedding = self.model(inference_tensor)


