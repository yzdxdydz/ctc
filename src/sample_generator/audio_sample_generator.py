import numpy as np
import wave
from typing import Sequence, Tuple

from .sample_generator import SampleGenerator
from ..utils import Vector, Comparable


class AudioSampleGenerator(SampleGenerator):
    def __init__(self,
                 data: Sequence[
                     Tuple[Sequence[Vector], Sequence[Comparable]]
                 ],
                 noise_rate: float = 0.05,
                 max_item_size: int = 1 << 8
                 ):

        self.noise_rate = noise_rate
        self.preprocess(data, max_item_size)

    def preprocess(self, data, max_item_size):
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        self.base_sample = []

        x, z = data[indices[0]]
        long_z = [phoneme for phoneme in z]
        long_x = x
        for i in indices[1:]:
            x, z = data[indices[i]]
            if x.shape[0] + long_x.shape[0] <= max_item_size:
                long_x = np.concatenate((
                    long_x, x
                ))
                long_z += z
            else:
                self.base_sample.append((long_x, long_z))
                long_x = x
                long_z = [phoneme for phoneme in z]
        self.base_sample.append((long_x, long_z))

    def generate_sample(self) -> Sequence[
        Tuple[Sequence[Vector], Sequence[Comparable]]
    ]:
        indices = np.arange(len(self.base_sample))
        np.random.shuffle(indices)

        sample = []
        for i in indices:
            x, z = self.base_sample[i]
            x += self.noise_rate * np.random.normal(0, 1, x.shape)
            sample.append((x, z))

        return sample
