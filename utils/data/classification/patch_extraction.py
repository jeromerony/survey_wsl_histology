from random import random, randint
from math import cos, sin, radians, ceil

__all__ = [
    'Grid',
    'NoOverlap',
    'NoOverlapRotation',
    'NoOverlapRotation90',
    'Overlap25',
    'Overlap25Rotation',
    'Overlap25Rotation90',
    'Overlap50',
    'Overlap50Rotation',
    'Overlap50Rotation90',
    'Overlap66',
    'Overlap66Rotation',
    'Overlap66Rotation90',
    'Overlap75',
    'Overlap75Rotation',
    'Overlap75Rotation90',
    'RandomParallel',
    'RandomParallelRotation',
    'RandomRotation',
    'RandomProperRotation'
]


class Grid:
    def __init__(self, image_size, patch_size, overlap):
        assert patch_size[0] <= image_size[0] and patch_size[1] <= image_size[1], \
            'Cannot extract patches of dimensions ({},{}) in images of dimensions ({},{})'.format(*patch_size,
                                                                                                  *image_size)
        self.image_size = image_size
        self.patch_size = patch_size
        n_h = int(((self.image_size[0]) - self.patch_size[0]) / (self.patch_size[0] * (1 - overlap)) + 1)
        n_w = int(((self.image_size[1]) - self.patch_size[1]) / (self.patch_size[1] * (1 - overlap)) + 1)
        self.patches_positions = []
        for h in range(n_h):
            for w in range(n_w):
                self.patches_positions.append((int(h * self.patch_size[0] * (1 - overlap)),
                                               int(w * self.patch_size[1] * (1 - overlap))))

    def __len__(self):
        return len(self.patches_positions)


class NoOverlap(Grid):
    def __init__(self, image_size, patch_size):
        super(NoOverlap, self).__init__(image_size, patch_size, 0)

    def __call__(self, image, patch_index):
        position = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch


class NoOverlapRotation(NoOverlap):
    def __init__(self, image_size, patch_size):
        super(NoOverlapRotation, self).__init__(image_size, patch_size)

    def __len__(self):
        return super(NoOverlapRotation, self).__len__() * 4

    def __call__(self, image, patch_index):
        index = patch_index % super(NoOverlapRotation, self).__len__()
        patch = super(NoOverlapRotation, self).__call__(image, index)
        return patch.rotate(random() * 360)


class NoOverlapRotation90(NoOverlap):
    def __init__(self, image_size, patch_size):
        super(NoOverlapRotation90, self).__init__(image_size, patch_size)
        new_patch_pos = []
        for patch_pos in self.patches_positions:
            for i in range(4):
                angle = i * 90
                new_patch_pos.append((patch_pos, angle))
        self.patches_positions = new_patch_pos

    def __call__(self, image, patch_index):
        position, angle = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch.rotate(angle)


class Overlap25(Grid):
    def __init__(self, image_size, patch_size):
        super(Overlap25, self).__init__(image_size, patch_size, 0.25)

    def __call__(self, image, patch_index):
        position = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch


class Overlap25Rotation(Overlap25):
    def __init__(self, image_size, patch_size):
        super(Overlap25Rotation, self).__init__(image_size, patch_size)

    def __len__(self):
        return super(Overlap25Rotation, self).__len__() * 4

    def __call__(self, image, patch_index):
        index = patch_index % super(Overlap25Rotation, self).__len__()
        patch = super(Overlap25Rotation, self).__call__(image, index)
        return patch.rotate(random() * 360)


class Overlap25Rotation90(Overlap25):
    def __init__(self, image_size, patch_size):
        super(Overlap25Rotation90, self).__init__(image_size, patch_size)
        new_patch_pos = []
        for patch_pos in self.patches_positions:
            for i in range(4):
                angle = i * 90
                new_patch_pos.append((patch_pos, angle))
        self.patches_positions = new_patch_pos

    def __call__(self, image, patch_index):
        position, angle = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch.rotate(angle)


class Overlap66(Grid):
    def __init__(self, image_size, patch_size):
        super(Overlap66, self).__init__(image_size, patch_size, 512 / 736)

    def __call__(self, image, patch_index):
        position = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch


class Overlap66Rotation(Overlap66):
    def __init__(self, image_size, patch_size):
        super(Overlap66Rotation, self).__init__(image_size, patch_size)

    def __len__(self):
        return super(Overlap66Rotation, self).__len__() * 4

    def __call__(self, image, patch_index):
        index = patch_index % super(Overlap66Rotation, self).__len__()
        patch = super(Overlap66Rotation, self).__call__(image, index)
        return patch.rotate(random() * 360)


class Overlap66Rotation90(Overlap66):
    def __init__(self, image_size, patch_size):
        super(Overlap66Rotation90, self).__init__(image_size, patch_size)
        new_patch_pos = []
        for patch_pos in self.patches_positions:
            for i in range(4):
                angle = i * 90
                new_patch_pos.append((patch_pos, angle))
        self.patches_positions = new_patch_pos

    def __call__(self, image, patch_index):
        position, angle = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch.rotate(angle)


class Overlap50(Grid):
    def __init__(self, image_size, patch_size):
        super(Overlap50, self).__init__(image_size, patch_size, 0.5)

    def __call__(self, image, patch_index):
        position = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch


class Overlap50Rotation(Overlap50):
    def __init__(self, image_size, patch_size):
        super(Overlap50Rotation, self).__init__(image_size, patch_size)

    def __len__(self):
        return super(Overlap50Rotation, self).__len__() * 4

    def __call__(self, image, patch_index):
        index = patch_index % super(Overlap50Rotation, self).__len__()
        patch = super(Overlap50Rotation, self).__call__(image, index)
        return patch.rotate(random() * 360)


class Overlap50Rotation90(Overlap50):
    def __init__(self, image_size, patch_size):
        super(Overlap50Rotation90, self).__init__(image_size, patch_size)
        new_patch_pos = []
        for patch_pos in self.patches_positions:
            for i in range(4):
                angle = i * 90
                new_patch_pos.append((patch_pos, angle))
        self.patches_positions = new_patch_pos

    def __call__(self, image, patch_index):
        position, angle = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch.rotate(angle)


class Overlap75(Grid):
    def __init__(self, image_size, patch_size):
        super(Overlap75, self).__init__(image_size, patch_size, 0.5)

    def __call__(self, image, patch_index):
        position = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch


class Overlap75Rotation(Overlap75):
    def __init__(self, image_size, patch_size):
        super(Overlap75Rotation, self).__init__(image_size, patch_size)

    def __len__(self):
        return super(Overlap75Rotation, self).__len__() * 4

    def __call__(self, image, patch_index):
        index = patch_index % super(Overlap75Rotation, self).__len__()
        patch = super(Overlap75Rotation, self).__call__(image, index)
        return patch.rotate(random() * 360)


class Overlap75Rotation90(Overlap75):
    def __init__(self, image_size, patch_size):
        super(Overlap75Rotation90, self).__init__(image_size, patch_size)
        new_patch_pos = []
        for patch_pos in self.patches_positions:
            for i in range(4):
                angle = i * 90
                new_patch_pos.append((patch_pos, angle))
        self.patches_positions = new_patch_pos

    def __call__(self, image, patch_index):
        position, angle = self.patches_positions[patch_index]
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch.rotate(angle)


class RandomParallel:
    def __init__(self, image_size, patch_size):
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_h, self.n_w = self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def __len__(self):
        return self.n_h * self.n_w

    def __call__(self, image, patch_index):
        position = (randint(0, self.image_size[0] - self.patch_size[0] - 1),
                    randint(0, self.image_size[0] - self.patch_size[0] - 1))
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        return patch


class RandomParallelRotation(RandomParallel):
    def __init__(self, image_size, patch_size):
        super(RandomParallelRotation, self).__init__(image_size, patch_size)

    def __len__(self):
        return super(RandomParallelRotation, self).__len__() * 4

    def __call__(self, image, patch_index):
        patch = super(RandomParallelRotation, self).__call__(image, patch_index)
        return patch.rotate(randint(0, 3) * 90)


class RandomRotation:
    def __init__(self, image_size, patch_size):
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_h, self.n_w = self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def __len__(self):
        return self.n_h * self.n_w * 8

    def __call__(self, image, patch_index):
        position = (randint(0, self.image_size[0] - self.patch_size[0] - 1),
                    randint(0, self.image_size[0] - self.patch_size[0] - 1))
        patch = image.crop((position[1], position[0],
                            position[1] + self.patch_size[1], position[0] + self.patch_size[0]))
        angle = random() * 360
        return patch.rotate(angle)


def margins(angle, h, w):
    theta = radians(angle % 90)
    h_margin = ceil(h * cos(theta) + w * sin(theta))
    w_margin = ceil(w * cos(theta) + h * sin(theta))
    return h_margin, w_margin


class RandomProperRotation(RandomRotation):
    def __init__(self, image_size, patch_size):
        super(RandomProperRotation, self).__init__(image_size, patch_size)

    def __call__(self, image, patch_index):
        angle = random() * 360

        h_margin, w_margin = margins(angle, self.patch_size[0], self.patch_size[1])
        # random coordinates of upper left corner
        x, y = randint(0, self.image_size[0] - h_margin - 1), randint(0, self.image_size[1] - w_margin - 1)
        patch = image.crop((y, x, y + w_margin, x + h_margin)).rotate(angle)
        x, y = int((h_margin - self.patch_size[0]) / 2), int((w_margin - self.patch_size[1]) / 2)
        patch = patch.crop((y, x, y + self.patch_size[1], x + self.patch_size[0]))

        return patch