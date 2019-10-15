import logging


logging.basicConfig(level=logging.INFO, format="[%(asctime)-15s %name-12s] %(message)s")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
