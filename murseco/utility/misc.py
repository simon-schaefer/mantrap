import random
import string


def random_string(length: int = 5):
    """Generate a random string of fixed given length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))
