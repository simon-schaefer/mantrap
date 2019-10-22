import logging


logging.basicConfig(level=logging.INFO, format="[%(asctime)-15s %(filename)-10s %(levelname)-8s] %(message)s")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.WARNING)
