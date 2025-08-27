  
class StandardScaler:
    def __init__(self):
        self.name = "Standard Scaler"
        self.mean_ = None
        self.scale_ = None
        
    def fit_transform(self, X):
  
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
  
        return (X - self.mean_) / self.scale_ 