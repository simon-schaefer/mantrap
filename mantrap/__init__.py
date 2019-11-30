import logging


logging.basicConfig(level=logging.DEBUG, format="[%(asctime)-15s %(filename)-15s %(levelname)-6s] %(message)s")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.WARNING)
