from sklearn import set_config

set_config(transform_output="pandas")

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer
)

SCALERS = {
    "standard": lambda: StandardScaler(),
    "minmax": lambda: MinMaxScaler(feature_range=(0, 1)),
    "robust": lambda: RobustScaler(),
    "maxabs": lambda: MaxAbsScaler(),
    "normalize": lambda: Normalizer(norm="l2"),
    "power": lambda: PowerTransformer(method="yeo-johnson"),
    "quantile": lambda: QuantileTransformer(output_distribution="normal"),
}