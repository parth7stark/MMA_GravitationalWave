import os
import uuid
import pathlib
import importlib
import torch.nn as nn
from datetime import datetime
from proxystore.store import Store
from mma_gw.compressor import *
from mma_gw.generator import GWGenerator
from mma_gw.config import ClientAgentConfig
from omegaconf import DictConfig, OmegaConf
from typing import Union, Dict, OrderedDict, Tuple, Optional
from mma_gw.logger import ClientAgentFileLogger

import h5py
import numpy as np

from mma_gw.model.GW_client_model import ClientModel

class ClientAgent:
    """
    Contain functions that client performs
    The `ClientAgent` should act on behalf of the FL client to:
    - load configurations received from the server `ClientAgent.load_config`
    - compute local embeddings  `ClientAgent.compute_embeddings`
    - prepare data for communication `ClientAgent.get_parameters`
    - get a unique client id for server to distinguish clients `ClientAgent.get_id`


    Users can overwrite any class method to add custom functionalities of the client agent.
    """
    def __init__(
        self, 
        client_agent_config: ClientAgentConfig = ClientAgentConfig()
    ) -> None:
        self.client_agent_config = client_agent_config
        self._create_logger()
        self._load_model()
        self._load_generator()
        self._load_compressor()
        self._load_proxystore()

    def load_config(self, config: DictConfig) -> None:
        """Load additional configurations provided by the server."""
        self.client_agent_config = OmegaConf.merge(self.client_agent_config, config)
        self._load_model()
        self._load_compressor()

    def get_id(self) -> str:
        """Return a unique client id for server to distinguish clients."""
        if not hasattr(self, 'client_id'):
            if hasattr(self.client_agent_config, "client_id"):
                self.client_id = self.client_agent_config.client_id
            else:
                self.client_id = str(uuid.uuid4())
        return self.client_id
    

    def compute_embeddings(self, inference_data) -> None:
        """Compute local embedding using the local data."""
        self.generator.compute_embeddings(inference_data)


    def get_parameters(self) -> Union[Dict, OrderedDict, bytes, Tuple[Union[Dict, OrderedDict, bytes], Dict]]:
        """
        Get local embeddings for sending to server
        Return parameters for communication
        """
        params = self.generator.get_parameters()
        params = {k: v.cpu() for k, v in params.items()}
        if isinstance(params, tuple):
            params, metadata = params
        else:
            metadata = None
        if self.enable_compression:
            params = self.compressor.compress_model(params)
        return self.proxy(params)[0] if metadata is None else (self.proxy(params)[0], metadata)
    

    def proxy(self, obj):
        """
        Create the proxy of the object.
        :param obj: the object to be proxied.
        :return: the proxied object and a boolean value indicating whether the object is proxied.
        """
        if self.enable_proxystore:
            return self.proxystore.proxy(obj), True
        else:
            return obj, False
        
    def clean_up(self) -> None:
        """Clean up the client agent."""
        if hasattr(self, "proxystore") and self.proxystore is not None:
            try:
                self.proxystore.close(clear=True)
            except:
                self.proxystore.close()

    def _create_logger(self):
        """
        Create logger for the client agent to log local training process.
        You can modify or overwrite this method to create your own logger.
        """
        if hasattr(self, "logger"):
            return
        kwargs = {}
        if not hasattr(self.client_agent_config, "generator_configs"):
            kwargs["logging_id"] = self.get_id()
            kwargs["file_dir"] = "./output"
            kwargs["file_name"] = "result"
        else:
            kwargs["logging_id"] = self.client_agent_config.generator_configs.get("logging_id", self.get_id())
            kwargs["file_dir"] = self.client_agent_config.generator_configs.get("logging_output_dirname", "./output")
            kwargs["file_name"] = self.client_agent_config.generator_configs.get("logging_output_filename", "result")
        if hasattr(self.client_agent_config, "experiment_id"):
            kwargs["experiment_id"] = self.client_agent_config.experiment_id
        self.logger = ClientAgentFileLogger(**kwargs)

    def _load_model(self) -> None:
        """
        Load client side HDCN model from the definition file (read the source code of the model)
        - Users can define their own way to load the model from other sources

        If checkpoint file directory is provided in config then load the checkpoints/best weights
        """

        model_configs =  self.client_agent_config.client_configs.model_configs

        self.model = ClientModel()

        if hasattr(self.client_agent_config.model_configs, "checkpoint_dir"):
            # load the checkpoint on the model created above
            client_weights = torch.load(self.client_agent_config.model_configs.checkpoint_dir)

            # Load weights into client model
            self.model.sub_mod.load_state_dict(client_weights)

    def _load_generator(self) -> None:
        """
        do what load_trainer is doing
        Load embeddings generator and initialize parameters
        """

        self.generator: GWGenerator = GWGenerator(
            model=self.model, 
            generator_configs=self.client_agent_config.generator_configs,
            logger=self.logger,
        )

        
    def _load_compressor(self) -> None:
        """
        Create a compressor for compressing the model parameters.
        """
        if hasattr(self, "compressor") and self.compressor is not None:
            return
        self.compressor = None
        self.enable_compression = False
        if not hasattr(self.client_agent_config, "comm_configs"):
            return
        if not hasattr(self.client_agent_config.comm_configs, "compressor_configs"):
            return
        if getattr(self.client_agent_config.comm_configs.compressor_configs, "enable_compression", False):
            self.enable_compression = True
            self.compressor = eval(self.client_agent_config.comm_configs.compressor_configs.lossy_compressor)(
               self.client_agent_config.comm_configs.compressor_configs
            )

    def _load_proxystore(self) -> None:
        """
        Create the proxystore for storing and sending the model parameters from the client to the server.
        """
        if hasattr(self, "proxystore") and self.proxystore is not None:
            return
        self.proxystore = None
        self.enable_proxystore = False
        if not hasattr(self.client_agent_config, "comm_configs"):
            return
        if not hasattr(self.client_agent_config.comm_configs, "proxystore_configs"):
            return
        if getattr(self.client_agent_config.comm_configs.proxystore_configs, "enable_proxystore", False):
            self.enable_proxystore = True
            from proxystore.connectors.redis import RedisConnector
            from proxystore.connectors.file import FileConnector
            from proxystore.connectors.endpoint import EndpointConnector
            from appfl.communicator.connector.s3 import S3Connector
            self.proxystore = Store(
                self.get_id(),
                eval(self.client_agent_config.comm_configs.proxystore_configs.connector_type)(
                    **self.client_agent_config.comm_configs.proxystore_configs.connector_configs
                ),
            )

    def load_inference_data(self, dataset_file) -> None:
        """Get local strain data for Inference."""

        data = h5py.File(dataset_file, 'r')
        
        GPS_start_time = int(data["GPS_start_time"][0])

        if self.client_agent_config.client_id=='0':
            return data['strain_L1'][:], GPS_start_time 
           
        elif self.client_agent_config.client_id=='1':
            return data['strain_H1'][:], GPS_start_time
            


