#%% Import required packages


#%%
class Visualizer:
    """
    A class representing a visualizer for the simulation. The visualizer is responsible for rendering the simulation
    environment and agents in a graphical format.

    Parameters
    ----------
    engine: Engine
        Reference to the simulation engine.

    Attributes
    ----------
    engine: Engine
        Reference to the simulation engine.
    """

    def __init__(self, engine):
        self.engine = engine

    def render(self):
        """
        Renders the simulation environment and agents. This method should be implemented by subclasses to provide
        specific visualization functionality.
        """
        raise NotImplementedError("The render method must be implemented by subclasses.")


