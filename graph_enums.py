import random
from enum import Enum
from typing import Tuple


class WeightDistribution(Enum):
    UNIFORM = "uniform"
    INTEGER = "integer"
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    
    def generate_params(self) -> Tuple:
        if self == WeightDistribution.UNIFORM:
            w_min = random.uniform(0.5, 3.0)
            return w_min, random.uniform(w_min + 2, w_min + 20)
        elif self == WeightDistribution.INTEGER:
            w_min = random.randint(1, 5)
            return w_min, random.randint(w_min + 10, w_min + 100)
        elif self == WeightDistribution.EXPONENTIAL:
            return (random.uniform(0.5, 3.0),)
        elif self == WeightDistribution.NORMAL:
            mean = random.uniform(3, 10)
            return mean, random.uniform(1, mean / 2)

    def generate_weight(self, params: Tuple) -> float:
        if self == WeightDistribution.UNIFORM:
            return random.uniform(params[0], params[1])
        elif self == WeightDistribution.INTEGER:
            return float(random.randint(int(params[0]), int(params[1])))
        elif self == WeightDistribution.EXPONENTIAL:
            return random.expovariate(params[0]) + 1.0
        elif self == WeightDistribution.NORMAL:
            return max(0.1, random.normalvariate(params[0], params[1]))

    @property
    def name_str(self) -> str:
        return self.value


class DensityTier(Enum):
    SPARSE = "sparse"
    MEDIUM = "medium"
    DENSE = "dense"
    
    def generate_edge_prob(self) -> float:
        if self == DensityTier.SPARSE:
            return random.uniform(0.05, 0.15)  # 5-15% of max edges
        elif self == DensityTier.MEDIUM:
            return random.uniform(0.15, 0.30)  # 15-30% of max edges
        elif self == DensityTier.DENSE:
            return random.uniform(0.30, 0.50)  # 30-50% of max edges
