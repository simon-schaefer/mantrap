from mantrap.agents.agent import Agent
from mantrap.agents.integrator import IntegratorDTAgent
from mantrap.agents.integrator_double import DoubleIntegratorDTAgent

AGENTS = [IntegratorDTAgent, DoubleIntegratorDTAgent]
AGENTS_DICT = {agent.agent_type(): agent for agent in AGENTS}
