import argparse
from omegaconf import OmegaConf
from mma_gw.agent import ClientAgent
from mma_gw.communicator.octopus import OctopusClientCommunicator

from inference_utils import *
from tqdm import tqdm
import glob
import time
import json


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="examples/configs/client1.yaml",
    help="Path to the configuration file."
)
args = argparser.parse_args()

# Load config from YAML (via OmegaConf)
client_agent_config = OmegaConf.load(args.config)

# Load HDCN with best weights
client_agent = ClientAgent(client_agent_config=client_agent_config)

# Create client-side communicator
client_communicator = OctopusClientCommunicator(
    client_agent,
    client_id = client_agent.get_id(),
    logger=client_agent.logger,

)

print(f"[Detector {client_agent.get_id()}] Waiting for ServerStarted event...", flush=True)
client_agent.logger.info(f"[Detector {client_agent.get_id()}] Waiting for ServerStarted event...")

# 1) Wait for ServerStarted event
for msg in client_communicator.consumer:
    client_agent.logger.info(f"[Detector {client_agent.get_id()}] msg: {msg}")

    data_str = msg.value.decode("utf-8")
    data = json.loads(data_str)

    Event_type = data["EventType"]

    if Event_type == "ServerStarted":        
        client_communicator.on_server_started(data)
        break  # We can break from the loop as we only need that single event
    

#  Start producing embeddings
print(f"[Detector {client_agent.get_id()}] ready for inference. Now computing embeddings and sending to server...")
client_agent.logger.info(f"[Detector {client_agent.get_id()}] ready for inference. Now computing embeddings and sending to server...")



if client_agent.client_agent_config.train_configs.do_inference==False:
    # Implement fine-tuning workflow using Octopus [TODO]

else:
    ### Inference workflow ###

    # Read the inference dataset (replaced glob, performing inference on only 1 hdf5 file)
    data_dir = client_agent.client_agent_config.inference_configs.dataset_path
    dataset_name = data_dir.split('/')[-1].split('_')[0]

    print(f"Performing inference on {dataset_name} dataset", flush=True)
    client_agent.logger.info(f"[Detector {client_agent.get_id()}] Performing inference on {dataset_name} dataset")

    # Load strains at the detector (local data)
    strain_data, GPSStartTime = client_agent.load_inference_data(data_dir)

    """ 
    Preprocess strain data
    - Normalize the strain
    - Trim the strain according to length paramter ([For testing purposes] Only take a certain length of strain for inference)
    """
    preprocessed_strain_data = preprocess(strain_data, client_agent.client_agent_config.inference_configs.length)

    # Create datagenerators
    dataloader_0 = TimeSeriesDataset(data=preprocessed_strain_data, targets=preprocessed_strain_data, length=4096, stride=4096, start_index=0)
    dataloader_5 = TimeSeriesDataset(data=preprocessed_strain_data, targets=preprocessed_strain_data, length=4096, stride=4096, start_index=2047)
    
    config_batch_size = client_agent.client_agent_config.inference_configs.batch_size
    dataloader_0 = DataLoader(dataloader_0, batch_size=config_batch_size, shuffle=False)
    dataloader_5 = DataLoader(dataloader_5, batch_size=config_batch_size, shuffle=False)
   

    """
    for batch in dataloader:
        client_agent.compute_embeddings(batch)
        local_embeddings = client_agent.get_embeddings()
        client_communicator.send_embeddings(local_embeddings)
    """
    start_time = time.time()
    print("Predicting 0")

    batch_no = 0
    for inputs in tqdm(dataloader_0, desc='Predicting 0', leave=True, disable=False):
        client_agent.compute_embeddings(inputs[0])
        local_embeddings = client_agent.get_parameters()
        
        """
        local_embeddings is a dictionary of format
            return {
            'inference_embedding': self.inference_embedding.cpu()
        }

        {'inference_embedding': tensor([[[ 4.6326,  4.5406,  4.4876,  ..., 25.6749, 24.8010, 24.3537],
        [ 6.2329,  6.2519,  6.2359,  ...,  6.8050,  6.2759,  5.6775],
        [ 5.3650,  5.1982,  5.2022,  ...,  9.6494,  8.9188,  7.9569],
        ...,

       """

        local_embedding_tensor = local_embeddings['inference_embedding']
        
        print("Data type of tensor: ", flush=True)
        print(local_embedding_tensor.dtype, flush=True)  

        print("Tensor shape: ", local_embedding_tensor.shape, flush=True )    
        
        client_communicator.send_embeddings_inference_Octopus(local_embedding_tensor, append_in="preds_0", batch_id = batch_no)
        batch_no = batch_no + 1
        
        # Async don't wait for acknowledgement --> keep on sending embeddings to evetn fabric

            
    print("Predicting 5")
    batch_no = 0
    for inputs in tqdm(dataloader_5, desc='Predicting 5', leave=True, disable=False):
        client_agent.compute_embeddings(inputs[0])
        local_embeddings = client_agent.get_parameters()
        local_embedding_tensor = local_embeddings['inference_embedding']
        
        client_communicator.send_embeddings_inference_Octopus(local_embedding_tensor, append_in="preds_5", batch_id = batch_no)
        batch_no = batch_no + 1
        
    
    elapsed_time = time.time() - start_time
    print(f"Time to make predictions: {elapsed_time:.2f} seconds", flush=True)
    

    print(f"[Detector {client_agent.get_id()}] Invoking post-process pipeline", flush=True)
    client_agent.logger.info(f"[Detector {client_agent.get_id()}] Invoking post-process pipeline")

    client_communicator.invoke_post_process(GPSStartTime)

    # trigger clean up function
    # close producer, consumer and clean up other things
