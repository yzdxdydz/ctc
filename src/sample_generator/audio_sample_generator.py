import numpy as np
from typing import Sequence, Tuple
from librosa import feature

from .sample_generator import SampleGenerator
from ..utils import Vector, Comparable


class AudioSampleGenerator(SampleGenerator):
    def __init__(self,
                 base_sample: Sequence[
                     Tuple[Sequence[Vector], Sequence[Comparable]]
                 ],
                 sr: int,
                 noise_rate: float = 0.05,
                 n_mfcc: int = 20,
                 ):

        # Amplitute normalization
        max_amp = max(np.max(np.abs(item[0])) for item in base_sample)
        self.base = []
        for item in base_sample:
            y = item[0] / max_amp
            y += noise_rate*np.random.normal(0, 1, y.shape)
            x = feature.mfcc(y=y,
                             sr=sr,
                             n_mfcc=n_mfcc).T
            self.base.append((x, item[1]))

    def generate_batches(self, batch_size: int) -> Sequence[
        Tuple[Sequence[Vector], Sequence[Comparable]]
    ]:
        indices = np.arange(len(self.base))
        np.random.shuffle(indices)

        sample = [self.base[i] for i in indices]

        batch_count = len(sample) // batch_size

        batches = [sample[i*batch_size:(i+1)*batch_size]
                   for i in range(batch_count)]

        if len(sample) % batch_size != 0:
            batches.append([sample[batch_count*batch_size:]])

        return batches
