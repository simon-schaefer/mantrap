from mantrap.agents.base.discrete import DTAgent
from mantrap.agents.integrator_single import IntegratorDTAgent
from mantrap.agents.integrator_double import DoubleIntegratorDTAgent

AGENTS = [IntegratorDTAgent, DoubleIntegratorDTAgent]
AGENTS_DICT = {agent.agent_type(): agent for agent in AGENTS}
