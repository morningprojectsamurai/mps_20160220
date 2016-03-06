from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from collections import OrderedDict
import test_images

class ScaleSpace(list):
    def __init__(self, orig_image, sigma=1.6, s=3, extra=2, min_width=16, min_height=16, *args):
        super().__init__(*args)
        self.orig_image = orig_image
        self.sigma = sigma
        self.s = s
        self.extra = extra
        self.min_width = min_width
        self.min_height = min_height

    def create(self, sigma=None, s=None, extra=None, min_width=None, min_height=None):
        sigma = sigma if sigma else self.sigma
        s = s if s else self.s
        k = np.power(2.0, 1.0/s)
        extra = extra if extra else self.extra
        min_width = min_width if min_width else self.min_width
        min_height = min_height if min_height else self.min_height

        def create_octaves(base_img):
            self.append(OrderedDict())
            self[-1][sigma] = base_img
            for i in range(1, s + extra + 1):
                scale = sigma * np.power(k, i)
                image = gaussian_filter(base_img, scale)
                self[-1][scale] = image
            if base_img.shape[0] / 2.0 > min_height and base_img.shape[1] > min_width:
                create_octaves(list(self[-1].values())[s][::2, ::2])
        base_img = gaussian_filter(self.orig_image, self.sigma)
        create_octaves(base_img)


class DoGSpace(list):
    def __init__(self, scale_space, *args):
        super().__init__(*args)
        self.scale_space = scale_space

    def create(self):
        for octave in self.scale_space:
            self.append(OrderedDict())
            scales = list(octave.keys())
            for i in range(len(scales[:-1])):
                self[-1][scales[i]] = octave[scales[i + 1]] - octave[scales[i]]


class ExtremaSpace(list):
    def __init__(self, dog_space, *args):
        super().__init__(*args)
        self.dog_space = dog_space
        self.df2_space = []

    def calc_second_diff(self, row, col, d, sd):
        '''二階微分の計算
        '''
        (d0, d1, d2) = d
        (sd0, sd1, sd2) = sd
        dxx = d0[row, col+2] - 2*d0[row, col+1] + d0[row, col]
        dyy = d0[row+2, col] - 2*d0[row+1, col] + d0[row, col]
        dzz = ((((d2[row, col] - d1[row,col])) / (sd2 - sd1)) \
               - ((d1[row, col] - d1[row, col]) / (sd1 - sd0)) ) \
            / (sd1 - sd0)
        dxy = (d0[row+1, col+1] - d1[row+1, col] - d0[row, col+1] + d0[row, col])\
              / (sd1 - sd0)
        dxz = (d1[row, col+1] - d1[row, col] - d0[row, col+1] + d0[row, col]) \
              / (sd1 - sd0)
        dyz = (d1[row+1, col] - d1[row, col] - d0[row+1, col] + d0[row, col]) \
              / (sd1 - sd0)
        return ( (dxx, dxy, dxz),
                 (dxy, dyy, dyz),
                 (dxz, dyz, dzz) )

    def localize(self):
        self.df2_space = []     # 結果はこの変数に保存
        for octave in self.dog_space:
            df2dict = OrderedDict()
            scales = list(octave.keys())
            for si in range(0, len(scales) - 2):
                sigma = (scales[si], scales[si+1], scales[si+2])
                d = (octave[sigma[0]], octave[sigma[1]], octave[sigma[2]])
                df2dict[sigma[0]] = []
                (height, width) = octave[sigma[0]].shape
                sdiff = [[[] for i in range(height-2)] for i in range(width-2)]
                for row in range(0, height - 2):
                    for col in range(0, width - 2):
                        sdiff[row][col] = self.calc_second_diff(row, col,d, sigma)
                df2dict[sigma[0]] = sdiff
            self.df2_space.append(df2dict)

    def find(self):
        def is_extremum(octave, scales, si, row, col):
            is_min = True
            is_max = True

            for _si in [si - 1, si, si + 1]:
                for _row in [row - 1, row,  row + 1]:
                    for _col in [col - 1, col, col + 1]:
                        if _si == si and _row == row and _col == col:
                            continue
                        if octave[scales[_si]][_row, _col] <= octave[scales[si]][row, col]:
                            is_min = False
                        if octave[scales[_si]][_row, _col] >= octave[scales[si]][row, col]:
                            is_max = False
                        if not is_min and not is_max:
                            return False
            return True

        for oi, octave in enumerate(dog_space):
            extrema = OrderedDict()
            scales = list(octave.keys())
            for si in range(1, len(scales) - 1):
                extrema[scales[si]] = []
                image = octave[scales[si]]
                for row in range(1, image.shape[0] - 1):
                    for col in range(1, image.shape[1] - 1):
                        if is_extremum(octave, scales, si, row, col):
                            extrema[scales[si]].append((row, col))
            self.append(extrema)


if __name__ == '__main__':
    test_image = Image.open('../img/lena.jpg').convert('L')
    #test_image = Image.open('img/lena.jpg').convert('L')
    #test_image = test_images.rectangle(50, 50, 50, 50)
    #test_image = test_images.ellipse(50, 50, 50, 100)
    test_image = np.array(test_image, dtype=np.float) / 255

    scale_space = ScaleSpace(test_image)
    scale_space.create()

    dog_space = DoGSpace(scale_space)
    dog_space.create()

    extrema_space = ExtremaSpace(dog_space)
    extrema_space.find()
    extrema_space.localize()

    f, ax = plt.subplots(len(dog_space), len(dog_space[0]))
    for i in range(len(dog_space)):
        for j, (scale, image) in enumerate(dog_space[i].items()):
            ax[i][j].set_xlim(0, image.shape[1])
            ax[i][j].set_ylim(image.shape[0], 0)
            ax[i][j].imshow(image, cmap='Greys_r')
            if j in range(1, len(dog_space[i]) - 1):
                ax[i][j].plot([coordinate[1] for coordinate in extrema_space[i][scale]], [coordinate[0] for coordinate in extrema_space[i][scale]], 'ro')
            ax[i][j].set_title(str(np.round(scale * np.power(2, i), 3)))
    plt.tight_layout()
    plt.show()

