import torch
from omegaconf import DictConfig
from typing import Any, Union, List, Dict, Optional
import numpy as np
from .GW_aggregator_utils import *
from lal import gpstime

class GWAggregator():
    """
    GWAggregator:
        Aggregator for vertical federated learning, which takes in local embeddings from clients,
        concatenates them using GNN, and trains a model on the concatenated embeddings. The aggregator then
        sends back the gradient of the loss with respect to the concatenated embeddings to the clients
        for them to update their local embedding models.
    """
    def __init__(
        self,
        model: torch.nn.Module | None = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Any | None = None,
    ):

        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        self.device = self.aggregator_configs.get("device", "cpu")
        self.model.to(self.device)
       
        
        
        # # Collect final predictions
        # Use dictionaries to store predictions in correct order
        # e.g., self.predictions_0[batch_id] = np.array([...]) 
        #       self.predictions_5[batch_id] = np.array([...])
        self.predictions_0 = {}
        self.predictions_5 = {}


        # Our local aggregator: store partial embeddings until we have both detectors.
        # aggregator[(batch_id, shift)] = {0: embedding0, 1: embedding1}
        self.aggregator = {}

        # Track which clients have finished
        self.completed_clients = set()
        self.expected_clients = { "0", "1" }  # Adjust as needed
        

    def process_embeddings_message(self, batch_id, shift, det_id, local_embedding):
        
        print(f"[Server] Received Embeddings: detector_id={det_id}, "
              f"batch_id={batch_id}, shift={shift}", flush=True)
        print(f"embedding.shape={tuple(local_embedding.shape)}", flush=True)

        self.logger.info(f"[Server] Received Embeddings: detector_id={det_id}, "
              f"batch_id={batch_id}, shift={shift}")
        
        key = (batch_id, shift)
        if key not in self.aggregator:
            self.aggregator[key] = {}

        self.aggregator[key][det_id] = local_embedding

        # Check if we have both detector embeddings for this (batch_id, shift)
        if "0" in self.aggregator[key] and "1" in self.aggregator[key]:
            local_embeddings = {
                "0": {"inference_embedding": self.aggregator[key]["0"]},
                "1": {"inference_embedding": self.aggregator[key]["1"]}
            }
            self.inference(local_embeddings, batch_id=batch_id, shift=shift)
            del self.aggregator[key]

    def inference(
        self, 
        local_embeddings: Dict[str, Dict], 
        **kwargs
    ):
        """
        Aggregate client embeddings for inference using a GNN-based server model.

        Args:
            local_embeddings (Dict[str, Dict]): Dictionary containing client embeddings.
                Example:
                {
                    'client_1' or "0": {'inference_embedding': x_A_infer},
                    'client_2' or "1": {'inference_embedding': x_B_infer}
                }
            append_in: Flag indicating whether to add the output in pred_0 or pred_5 list
        Returns:
            None
        """

        # Extract embeddings from each client - the client_Id should be 0 or 1 and not client_1 and client_2. Otherwise below code will fail
        x_A_infer = local_embeddings['0']['inference_embedding']
        x_B_infer = local_embeddings['1']['inference_embedding']
        
        x_A_infer = x_A_infer.to(self.device)
        x_B_infer = x_B_infer.to(self.device)

        append_in =  kwargs["shift"]
        batch_id = kwargs["batch_id"]

        with torch.no_grad():
            self.model.eval()
            outputs = self.model(x_A_infer, x_B_infer)
            
            if append_in=='preds_0':
                self.predictions_0[batch_id]=(outputs.detach().cpu().numpy())
            else:
                self.predictions_5[batch_id]=(outputs.detach().cpu().numpy())

    def process_post_process_message(self, producer, topic, det_id, status, GPS_start_time):
        """
        Handle "PostProcess" messages. Wait until all clients are done.
        """    
        if status == "DONE" and det_id is not None:
            print(f"[Server] Received DONE from detector {det_id}")
            self.logger.info(f"[Server] Received DONE from detector {det_id}")

            # Add the client to the completed set
            self.completed_clients.add(det_id)

            # Check if all expected clients are done
            if self.completed_clients == self.expected_clients:
                print("[Server] All detectors are DONE. Invoking post_process...")
                self.logger.info("[Server] All detectors are DONE. Invoking post_process...")
                self.post_process(producer, topic, GPS_start_time)

    def post_process(self, producer, topic, GPS_start_time):
        """
        Once both detectors are done, we gather predictions in ascending order of batch_id.
        """
        if self.predictions_0:
            # Sort by batch_id
            sorted_ids_0 = sorted(self.predictions_0.keys())
            # Gather in order
            preds_list_0 = [self.predictions_0[bid] for bid in sorted_ids_0]
            preds_0 = np.concatenate(preds_list_0, axis=0).ravel()
        else:
            preds_0 = np.array([])

        if self.predictions_5:
            sorted_ids_5 = sorted(self.predictions_5.keys())
            preds_list_5 = [self.predictions_5[bid] for bid in sorted_ids_5]
            preds_5 = np.concatenate(preds_list_5, axis=0).ravel()
        else:
            preds_5 = np.array([])

        print(f"[Server] Final preds_0 shape: {preds_0.shape}", flush=True)  #length of signal = length of output tensor
        print(f"[Server] Final preds_5 shape: {preds_5.shape}", flush=True)

        
        width = self.aggregator_configs.post_process_configs.width
        threshold = self.aggregator_configs.post_process_configs.threshold

        triggers = get_triggers(preds_0, preds_5, width, threshold, truncation=0, window_shift=2048)
       
        """
        triggers is a dictionary of format
        triggers = {
            'detection': [comma separated time list]
        }

        When multiple datasets given, triggers is dictionary keyed on dataset name
        Dataset Triggers =
        {
            GW200129: [{'detection': [1925.3505859375]}]
            GW200219: [{'detection': []}]
            GW200208: [{'detection': [812.976806640625, 932.83935546875, 1483.5166015625, 1162.1025390625, 2237.10302734375, 3679.439208984375]}]
        }

        """

        # Check if we have any detection
        if 'detection' in triggers and triggers['detection']:
            print(f"triggers: {triggers}" , flush=True)
            self.logger.info(f"triggers: {triggers}")

            # If we have detection, send merger details to Octopus
            self.publish_detection_details(producer, topic, triggers, GPS_start_time)
        else:
            # Log that triggers are either empty
            print(f"No detection triggers yet", flush=True)
            self.logger.info("No detection triggers yet")
                

    def publish_detection_details(self, producer, topic, triggers, GPS_start_time):
       
        # Compute GPS detection times
        gps_detection_times = [GPS_start_time + t for t in triggers['detection']]
        print(f"GPS start time: {GPS_start_time}", flush=True)
        self.logger.info(f"GPS start time: {GPS_start_time}")


        # Convert GPS detection times to UTC times
        utc_detection_times = [gpstime.gps_to_utc(gps_time) for gps_time in gps_detection_times]

        # Prepare data to send to Kafka
        detection_details = []
        for gps_time, utc_time in zip(gps_detection_times, utc_detection_times):
            print(f"GPS Time: {gps_time} -> UTC Time: {utc_time}", flush=True)
            self.logger.info(f"GPS Time: {gps_time} -> UTC Time: {utc_time}")
            
            detection_detail = {
                "GPS_time": gps_time,
                "UTC_time": utc_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            detection_details.append(detection_detail)

        # Send detection details to Kafka
        producer.send(topic, value={
        
            "EventType": "PotentialMerger",
            "detection_details": detection_details
        })
        producer.flush()
        
        print("[Server] Published PotentialMerger event with GPS time.", flush=True)
        self.logger.info("[Server] Published PotentialMerger event with GPS time.")
