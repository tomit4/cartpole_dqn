from collections import namedtuple

# Transition tuple for replay buffer (s, a, s', r)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
