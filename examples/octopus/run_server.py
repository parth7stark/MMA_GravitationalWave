import argparse
from omegaconf import OmegaConf
from mma_gw.agent import ServerAgent
from mma_gw.communicator.octopus import OctopusServerCommunicator
import json


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", 
    type=str, 
    default="examples/configs/server.yaml",
    help="Path to the configuration file."
)

args = argparser.parse_args()

# Load config from YAML (via OmegaConf)
server_agent_config = OmegaConf.load(args.config)

# Load GNN with best weights
server_agent = ServerAgent(server_agent_config=server_agent_config)

# Create server-side communicator
communicator = OctopusServerCommunicator(
    server_agent,
    logger=server_agent.logger,
)


# Publish "ServerStarted" event with config so that clients can pick it up
communicator.publish_server_started_event()


print("[Server] Listening for messages...", flush=True)
server_agent.logger.info("[Server] Listening for messages...")

for msg in communicator.consumer:
    topic = msg.topic
    data_str = msg.value.decode("utf-8")  # decode to string
    data = json.loads(data_str)          # parse JSON to dict

    Event_type = data["EventType"]

    if Event_type == "SendEmbeddings":
        communicator.handle_embeddings_message(data)

    elif Event_type == "PostProcess":
        communicator.handle_post_process_message(data)

    elif Event_type == "PotentialMerger":
        # not triggering anything on server side
        continue

    elif Event_type == "DetectorReady":  
        # Detector connected and ready for inference
        # not triggering anything on server side, just publishing event to octopus fabric
        # Keep on listening other events
        continue 

        # Later we will keep track of connected detectors and check if anyone got disconnected

    elif Event_type == "ServerStarted":
        # Continue listening other events
        continue

    else:
        print(f"[Server] Unknown Event Type in topic ({topic}): {Event_type}", flush=True)
        server_agent.logger.info(f"[Server] Unknown Event Type in topic ({topic}): {Event_type}")