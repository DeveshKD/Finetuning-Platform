"""One-line logging config shared by the API, orchestrator, and worker
processes so log lines are consistently formatted regardless of which
process emitted them.
"""

import logging


def configure_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
