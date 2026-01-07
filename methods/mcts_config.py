"""
Configuration settings for MCTS + LLM Agent.

This module provides configurable parameters for fine-tuning the MCTS algorithm.
Different scenarios may require different settings for optimal performance.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MCTSConfig:
    """
    Configuration for MCTS algorithm.
    
    Attributes:
        simulations: Number of MCTS simulations per move (higher = stronger but slower)
        exploration_weight: UCB exploration constant (higher = more exploration)
        max_depth: Maximum search depth per simulation (prevents infinite loops)
        time_limit: Time limit per move in seconds (None = no limit)
        use_llm_eval: Whether to use LLM for position evaluation
        eval_temperature: Temperature for LLM evaluation (lower = more deterministic)
        eval_max_tokens: Maximum tokens for LLM evaluation response
        eval_timeout: Timeout for LLM API calls in seconds
        prune_threshold: Minimum distance to existing stones to consider a move
        max_children: Maximum number of child nodes to explore
        cache_size: Maximum size of evaluation cache (0 = unlimited)
        verbose: Whether to print detailed progress information
    """
    
    # MCTS parameters
    simulations: int = 50
    exploration_weight: float = 1.41  # sqrt(2)
    max_depth: int = 20
    time_limit: Optional[float] = None
    
    # LLM evaluation parameters
    use_llm_eval: bool = True
    eval_temperature: float = 0.1
    eval_max_tokens: int = 10
    eval_timeout: int = 30
    
    # Search optimization parameters
    prune_threshold: int = 2  # Consider moves within N cells of existing stones
    max_children: int = 20    # Maximum candidate moves to explore
    cache_size: int = 1000    # Maximum cached evaluations
    
    # Logging
    verbose: bool = True
    log_interval: int = 10    # Log progress every N simulations


# Predefined configurations for different scenarios

FAST_CONFIG = MCTSConfig(
    simulations=20,
    exploration_weight=1.41,
    max_depth=15,
    use_llm_eval=True,
    prune_threshold=2,
    max_children=15,
    verbose=True,
    log_interval=10,
)
"""Fast configuration: Quick decisions, suitable for testing and rapid games."""

BALANCED_CONFIG = MCTSConfig(
    simulations=50,
    exploration_weight=1.41,
    max_depth=20,
    use_llm_eval=True,
    prune_threshold=2,
    max_children=20,
    verbose=True,
    log_interval=10,
)
"""Balanced configuration: Good trade-off between strength and speed."""

STRONG_CONFIG = MCTSConfig(
    simulations=100,
    exploration_weight=1.0,  # Less exploration, more exploitation
    max_depth=30,
    use_llm_eval=True,
    prune_threshold=3,
    max_children=30,
    verbose=True,
    log_interval=20,
)
"""Strong configuration: Maximum strength, suitable for competitive play."""

TOURNAMENT_CONFIG = MCTSConfig(
    simulations=150,
    exploration_weight=1.0,
    max_depth=40,
    time_limit=30.0,  # 30 seconds per move
    use_llm_eval=True,
    prune_threshold=3,
    max_children=40,
    verbose=False,  # Quiet mode for tournaments
    log_interval=30,
)
"""Tournament configuration: For serious competition with time controls."""

# Configuration without LLM (pure MCTS with random rollouts)
PURE_MCTS_CONFIG = MCTSConfig(
    simulations=200,  # Need more simulations without LLM guidance
    exploration_weight=1.41,
    max_depth=20,
    use_llm_eval=False,  # Use random playouts instead of LLM
    prune_threshold=2,
    max_children=25,
    verbose=True,
    log_interval=20,
)
"""Pure MCTS configuration: Traditional MCTS without LLM evaluation."""


def get_config_by_name(name: str) -> MCTSConfig:
    """
    Get a predefined configuration by name.
    
    Args:
        name: Configuration name ('fast', 'balanced', 'strong', 'tournament', 'pure')
    
    Returns:
        MCTSConfig instance
    
    Raises:
        ValueError: If configuration name is not recognized
    """
    configs = {
        'fast': FAST_CONFIG,
        'balanced': BALANCED_CONFIG,
        'strong': STRONG_CONFIG,
        'tournament': TOURNAMENT_CONFIG,
        'pure': PURE_MCTS_CONFIG,
    }
    
    name_lower = name.lower()
    if name_lower not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Unknown config name: {name}. Available: {available}")
    
    return configs[name_lower]


def print_config(config: MCTSConfig):
    """Print configuration settings in a readable format."""
    print("\n" + "=" * 60)
    print("MCTS Configuration")
    print("=" * 60)
    print(f"Simulations per move:    {config.simulations}")
    print(f"Exploration weight:      {config.exploration_weight}")
    print(f"Max search depth:        {config.max_depth}")
    print(f"Time limit:              {config.time_limit if config.time_limit else 'None'}")
    print(f"Use LLM evaluation:      {config.use_llm_eval}")
    if config.use_llm_eval:
        print(f"  - Temperature:         {config.eval_temperature}")
        print(f"  - Max tokens:          {config.eval_max_tokens}")
        print(f"  - Timeout:             {config.eval_timeout}s")
    print(f"Prune threshold:         {config.prune_threshold} cells")
    print(f"Max children:            {config.max_children}")
    print(f"Cache size:              {config.cache_size}")
    print(f"Verbose logging:         {config.verbose}")
    print("=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    print("Available MCTS Configurations:\n")
    
    for name in ['fast', 'balanced', 'strong', 'tournament', 'pure']:
        config = get_config_by_name(name)
        print(f"\n{name.upper()} Configuration:")
        print(f"  Simulations: {config.simulations}")
        print(f"  LLM Eval: {config.use_llm_eval}")
        print(f"  Est. time/move: ", end="")
        if name == 'fast':
            print("~5 seconds")
        elif name == 'balanced':
            print("~10-15 seconds")
        elif name == 'strong':
            print("~25-35 seconds")
        elif name == 'tournament':
            print("~30 seconds (time limited)")
        else:
            print("~40-60 seconds")




