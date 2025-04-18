#Version 1.1.1 conform as of 18.04.2025
"""
| *alts.modules.oracle.query_queue*
| :doc:`Core Module </core/oracle/query_queue>`
"""
from dataclasses import dataclass
from alts.core.oracle.query_queue import QueryQueue

@dataclass
class FCFSQueryQueue(QueryQueue):
    """
    FCFSQueryQueue()
    | **Description**
    |   The First Come First Serve Query Queue works like a queue, the first added queries are the first ones to be popped!
    """
    pass