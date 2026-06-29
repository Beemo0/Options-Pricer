from Option_type.BaseOption import BaseOption


class AmericanOption(BaseOption):
    """
    Option américaine : exercice possible à tout moment avant maturité.
    Compatible avec : Binomial, Trinomial, Différences finies.
    Non compatible avec Black-Scholes analytique.
    """

    def __init__(self, S, K, t, r, sigma, y=0.0):
        super().__init__(S, K, t, r, sigma, y)
        self.is_american = True

    def __repr__(self):
        return f"AmericanOption({super().__repr__()})"