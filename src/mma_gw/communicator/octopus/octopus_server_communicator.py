import json
import logging
from typing import Optional
from omegaconf import OmegaConf
from proxystore.proxy import extract
from mma_gw.agent import ServerAgent
from mma_gw.logger import ServerAgentFileLogger
from .utils import serialize_tensor_to_base64, deserialize_tensor_from_base64
from lal import gpstime

from diaspora_event_sdk import KafkaProducer, KafkaConsumer

class OctopusServerCommunicator:
    """
    Octopus communicator for federated learning server.
    Contains functions to produce/consume/handle different events
    """

    def __init__(
        self,
        server_agent: ServerAgent,
        logger: Optional[ServerAgentFileLogger] = None,
    ) -> None:

        self.server_agent = server_agent
        self.logger = logger if logger is not None else self._default_logger()

        self.topic = self.server_agent.server_agent_config.server_configs.comm_configs.octopus_configs.topic


        # Kafka producer for publishing messages
        self.producer = KafkaProducer()

        # Kafka consumer to listen for control events AND embeddings
        self.consumer = KafkaConsumer(
            self.topic,
            enable_auto_commit=True,
            group_id=self.server_agent.server_agent_config.server_configs.comm_configs.octopus_configs.group_id
        )

        # Track readiness, which detector is connected and ready for inference
        self.detectors_ready = set()

        # A flag indicating whether we should handle embeddings or not
        # If you want to handle embeddings as soon as the first client is ready, set True upon first client ready
        self.aggregator_active = False

        # This is to handle scenario: You do not wait for all detectors to say “ready” if you want to begin listening to embeddings as soon as any client is ready.

    def publish_server_started_event(self):
        """
        Publishes an event to the control topic indicating that the server has started,
        along with the configuration shared among all clients.
        """
        client_config =  self.server_agent.get_client_configs()
        client_config_dict = OmegaConf.to_container(client_config, resolve=True)

        event = {
            "EventType": "ServerStarted",
            "detector_config": client_config_dict,
        }

        self.producer.send(self.topic, value=event)
        self.producer.flush()
        
        print("[Server] Published ServerStarted event with config.", flush=True)
        self.logger.info("[Server] Published ServerStarted event with config.")

    def handle_embeddings_message(self, data):
        """
        Message of type "SendEmbeddings" is detected/consumed. Handle it
        Example of Message
        msg:  ConsumerRecord(topic='mma-GWwave-Triggers', partition=0, offset=0, timestamp=1736989957681, timestamp_type=0, key=None, value=b'{"EventType": "SendEmbeddings", "detector_id": "0", "batch_id": 0, "shift": "preds_0", "embedding": "UEsDBAAACAg
        """

        # Extract metadata
        batch_id = data["batch_id"]
        shift = data["shift"]           # "preds_0" or "preds_5"
        det_id = data["detector_id"]    # 0 or 1
        
        # Extract and deserialize the embedding
        embedding_b64 = data["embedding"]
        local_embedding = deserialize_tensor_from_base64(embedding_b64)

        self.server_agent.aggregator.process_embeddings_message(batch_id, shift, det_id, local_embedding)
    
    def handle_post_process_message(self, data):
        """
        Message of type "PostProcess" is detected/consumed. Handle it
        Example of Message
        msg:  ConsumerRecord(topic='mma-GWwave-Triggers', partition=0, offset=7705, timestamp=1736957074944, timestamp_type=0, key=None, value=b'{"EventType": "PostProcess", "detector_id": "1", "status": "DONE", "details": "DONE -> Invoke post process pipeline", "GPS_start_time": 1264314069}', headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=147, serialized_header_size=-1)
        """

        det_id = data["detector_id"]    # 0 or 1
        status = data["status"]
        GPS_start_time = data["GPS_start_time"]

        triggers = self.server_agent.aggregator.process_post_process_message(det_id, status, GPS_start_time)
    
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
            self.publish_detection_details(triggers, GPS_start_time)

            

    def publish_detection_details(self, triggers, GPS_start_time):
       
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
        self.producer.send(self.topic, value={
        
            "EventType": "PotentialMerger",
            "detection_details": detection_details
        })
        self.producer.flush()
        
        print("[Server] Published PotentialMerger event with GPS time.", flush=True)
        self.logger.info("[Server] Published PotentialMerger event with GPS time.")


    def _default_logger(self):
        """Create a default logger for the server if no logger provided."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s %(levelname)-4s server]: %(message)s')
        s_handler = logging.StreamHandler()
        s_handler.setLevel(logging.INFO)
        s_handler.setFormatter(fmt)
        logger.addHandler(s_handler)
        return logger


