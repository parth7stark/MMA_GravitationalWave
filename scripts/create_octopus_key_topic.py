"""
Diaspora Event Fabric offers topic-level access control. Typically, only one user can access a topic. Use register_topic API to request access (Replace ... below with your topic name):
"""

from diaspora_event_sdk import Client as GlobusClient
c = GlobusClient()
topic = ... # e.g., "topic-" + c.subject_openid[-12:]

print(c.register_topic(topic))
print(c.list_topics())