@dataclass
class RobustnessParams:
    """
    This class contains hyperparameters for measuring example robustness.

    -forget_thres: how many times an example needs to be forgotten to be recorded (default 3)
    -granularity: when to log robustness, either 'by_iter' or 'by_ep' (default by iteration)
    
    """

    forget_thres: int = 3
    granularity: str = 'by_iter'
    avg_over_inits: bool = True
    randomize_batches: bool = False