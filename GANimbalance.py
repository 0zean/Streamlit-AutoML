import pandas as pd
from imblearn.over_sampling.base import BaseOverSampler

from GANsample import GAN


class GanResample(BaseOverSampler):
    def __init__(self, sampling_strategy='auto', random_state=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        
    def _fit_resample(self, X, y):
        x_cols = list(X.columns)
        y_col = y.name
        
        gan = GAN(num_epochs=200,
          batch_size=32,
          d_hidden_dim=(768, 512, 384, 256),
          g_hidden_dim=(1024, 768, 512, 384, 256, 128),
          n_input=len(x_cols + y_col),
          stddev=0.15,
          d_learning_rate=0.001,
          g_learning_rate=0.00075,
          pretrain=False,
          random_state=42)
        
        Xf = X.join(y)
        
        gan.fit(Xf.loc[Xf[y_col] == 1, x_cols + y_col].values, display_step=1)
        
        synth_samples = pd.DataFrame(gan.sample(round((len(Xf.index) - sum(Xf[y_col]))/2)), columns = x_cols + y_col)
        
        over = synth_samples.loc[synth_samples[y_col] == 1, x_cols + y_col]

        new = pd.concat([Xf, over])
        new = new.sample(frac = 1)
        
        X_resampled = new.iloc[:,:-1]
        y_resampled = new[y_col]
        
        return X_resampled, y_resampled