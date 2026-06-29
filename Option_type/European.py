from Option_type.BaseOption import BaseOption


class EuropeanOption(BaseOption):
    """
    Option européenne : exercice uniquement à maturité.
    Compatible avec : Black-Scholes, Binomial, Trinomial, Monte Carlo, Différences finies.
    """

    def __init__(self, S, K, t, r, sigma, y=0.0):
        super().__init__(S, K, t, r, sigma, y)
        self.is_american = False

    def __repr__(self):
        return f"EuropeanOption({super().__repr__()})"