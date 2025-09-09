import logging
import proxystore
import proxystore.connectors
import os


def get_proxystore_connector(connector_name, connector_args, logger):
    assert connector_name in [
        "RedisConnector",
        "FileConnector",
        "EndpointConnector",
    ], f"Invalid connector name: {connector_name}, only RedisConnector, FileConnector, and EndpointConnector are supported"
    if connector_name == "RedisConnector":
        from proxystore.connectors.redis import RedisConnector

        connector = RedisConnector(**connector_args)
    elif connector_name == "FileConnector":
        from proxystore.connectors.file import FileConnector

        connector = FileConnector(**connector_args)
    elif connector_name == "EndpointConnector":
        from proxystore.connectors.endpoint import EndpointConnector

        endpoints = connector_args.pop("endpoints")
        for i, e in enumerate(endpoints):
            if "PROXYSTORE" in e:
                endpoints[i] = os.getenv(e, e)

        connector_args["endpoints"] = endpoints
        logger.error(f"CONNECTOR ARGS: {connector_args}")
        connector = EndpointConnector(**connector_args)
    return connector
