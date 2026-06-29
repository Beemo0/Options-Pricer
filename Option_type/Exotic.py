from Option_type.BaseOption import BaseOption


class AsianOption(BaseOption):
    """
    Option asiatique : le payoff dépend de la moyenne du sous-jacent.
    Paramètres supplémentaires :
    - averaging : 'arithmetic' ou 'geometric'
    - avg_type  : 'price' (sur le prix moyen) ou 'strike' (strike = moyenne)
    Uniquement priceable par Monte Carlo.
    """

    def __init__(self, S, K, t, r, sigma, y=0.0,
                 averaging="arithmetic", avg_type="price"):
        super().__init__(S, K, t, r, sigma, y)
        self.is_american = False
        self.averaging = averaging   # 'arithmetic' ou 'geometric'
        self.avg_type = avg_type     # 'price' ou 'strike'


class BarrierOption(BaseOption):
    """
    Option à barrière.
    Paramètres supplémentaires :
    - barrier      : niveau de la barrière
    - barrier_type : 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    Priceable par Monte Carlo ou Différences finies.
    """

    def __init__(self, S, K, t, r, sigma, y=0.0,
                 barrier=None, barrier_type="down-and-out"):
        super().__init__(S, K, t, r, sigma, y)
        self.is_american = False
        self.barrier = barrier
        self.barrier_type = barrier_type

        if barrier is None:
            raise ValueError("Une barrière doit être spécifiée pour une BarrierOption.")
        valid_types = {"up-and-out", "up-and-in", "down-and-out", "down-and-in"}
        if barrier_type not in valid_types:
            raise ValueError(f"barrier_type doit être parmi {valid_types}.")


class LookbackOption(BaseOption):
    """
    Option lookback : le payoff dépend du min/max historique.
    Paramètres supplémentaires :
    - lookback_type : 'fixed' (strike fixé) ou 'floating' (strike = min/max)
    Uniquement priceable par Monte Carlo.
    """

    def __init__(self, S, K, t, r, sigma, y=0.0, lookback_type="floating"):
        super().__init__(S, K, t, r, sigma, y)
        self.is_american = False
        self.lookback_type = lookback_type

        if lookback_type not in {"fixed", "floating"}:
            raise ValueError("lookback_type doit être 'fixed' ou 'floating'.")