class BaseOption:
    """Classe de base pour toutes les options."""

    def __init__(self, S, K, t, r, sigma, y=0.0):
        self.S = S          # Prix du sous-jacent
        self.K = K          # Strike
        self.t = t          # Maturité (en années)
        self.r = r          # Taux sans risque
        self.sigma = sigma  # Volatilité implicite
        self.y = y          # Taux de dividende continu

    def pricing_method(self, method, **kwargs):
        return method(self, **kwargs)

    def __repr__(self):
        return (f"{self.__class__.__name__}(S={self.S}, K={self.K}, "
                f"t={self.t}, r={self.r}, sigma={self.sigma}, y={self.y})")