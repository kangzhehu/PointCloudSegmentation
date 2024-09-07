import numpy as np


class RandomRotate:
    def __call__(self, points, labels, **kwargs):
        angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), -np.cos(angle), 0],
                                    [0, 0, 1]])
        points[:, :3] = points[:, :3] @ rotation_matrix
        return points, labels


class RandomJitter:
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points, labels, **kwargs):
        jittered_data = np.clip(self.sigma * np.random.randn(*points.shape), -self.clip, self.clip)
        points[:, :3] = points[:, :3] + jittered_data[:, :3]
        return points, labels


class RandomScale:
    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, points, labels, **kwargs):
        scale = np.random.uniform(self.scale_low, self.scale_high)
        points[:, :3] = points[:, :3] * scale
        return points, labels


class RandomFlip:
    def __init__(self, axis=0, prob=0.5):
        self.axis = axis,
        self.prob = prob

    def __call__(self, points, labels, **kwargs):
        if np.random.rand() < self.prob:
            points[:, self.axis] = -points[:, self.axis]
        return points, labels


class RandomTranslate:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, points, labels, translate_range=0.1, **kwargs):
        if np.random.rand() < self.prob:
            translation = np.random.uniform(-translate_range, translate_range, 3)
            points[:, :3] = points[:, :3] + translation
        return points, labels


class RandomShear:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, points, labels, shear_range=0.2, **kwargs):
        if np.random.rand() < self.prob:
            shear_matrix = np.eye(3) + np.random.uniform(-shear_range, shear_range, (3, 3))
            points[:, :3] = points[:, :3] @ shear_matrix
        return points, labels


class RandomNoise:
    def __init__(self, noise_level=0.01, prob=0.5):
        self.prob = prob
        self.noise_level = noise_level

    def __call__(self, points, labels, **kwargs):
        if np.random.rand() < self.prob:
            noise = np.random.normal(0, self.noise_level, points[:, :3].shape)
            points[:, :3] += noise
        return points, labels


class RandomDropout:
    def __init__(self, dropout_ratio=0.05, prob=0.5):
        self.prob = prob
        self.dropout_ratio = dropout_ratio

    def __call__(self, points, labels, **kwargs):
        if np.random.rand() < self.prob:
            num_points = points.shape[0]
            drop_mask = np.random.rand(num_points) > self.dropout_ratio
            if len(np.where(~drop_mask)[0]) > 0:  # 确保有点被丢弃
                points[~drop_mask, :] = points[0, :]  # 将被丢弃的点设置为第一个点
                labels[~drop_mask] = labels[0]  # 同样处理标签
        return points, labels


class ShuffleIdx:
    def __call__(self, points, labels, **kwargs):
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        return points[idx, ...], labels[idx]


class NormalizeCoordinates:
    def __call__(self, points, labels, **kwargs):
        max_value = points[:, :3].max(axis=0)
        points[:, 6] = points[:, 0] / max_value[0]
        points[:, 7] = points[:, 1] / max_value[1]
        points[:, 8] = points[:, 2] / max_value[2]

        return points, labels


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, labels):
        for t in self.transforms:
            points, labels = t(points, labels)
        return points, labels

