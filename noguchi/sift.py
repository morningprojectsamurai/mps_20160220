from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from collections import OrderedDict
#import test_images


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

    def localize(self):
        # ここにコードを書く
        pass

    # first derivative vector
    def dx(self, i, j, sigma, dsigma, octave):
        dx = octave[sigma][i, j+1] - octave[sigma][i, j]
        dy = octave[sigma][i+1, j] - octave[sigma][i, j]
        ds = (octave[sigma+dsigma][i, j] - octave[sigma][i, j]) / dsigma
        return np.array([dx, dy, ds])[:, np.newaxis]

    # second derivative matrix
    def d2fdx2(self, i, j, sigma, dsigma1, dsigma2, octave):
        # second derivatives
        d2x = octave[sigma][i, j+1] + octave[sigma][i, j-1] - 2 * octave[sigma][i, j]
        d2y = octave[sigma][i+1, j] + octave[sigma][i-1, j] - 2 * octave[sigma][i, j]
        d2s = ((octave[sigma+dsigma2][i, j] - octave[sigma][i, j])/dsigma2 - (octave[sigma][i, j] \
              - octave[sigma-dsigma1][i,j])/dsigma1)/dsigma1

        # cross
        d2xy = (octave[sigma][i+1, j+1] - octave[sigma][i+1, j]) - (octave[sigma][i, j+1] - octave[sigma][i,j])
        d2xs = (octave[sigma+dsigma2][i, j+1] - octave[sigma+dsigma2][i, j]) \
               - (octave[sigma][i, j+1] - octave[sigma][i, j]) / dsigma2

        d2ys = (octave[sigma+dsigma2][i+1, j] - octave[sigma+dsigma2][i, j]) \
               - (octave[sigma][i+1, j] - octave[sigma][i, j]) / dsigma2

        return np.array([[d2x, d2xy, d2xs],
                        [d2xy, d2y, d2ys],
                        [d2xs, d2ys, d2s]])

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

    def findByNoguchi(self, threshold = 0.00003):

        # (画像内の)極大値かを判定
        def isLocalMax(target, rng):
            r = np.array(rng)
            r[1][1][1] = -1 # 自分自身はmaxから除外する
            max_value = np.max(r)
            if target > max_value:
                return True
            else:
                return False

        # (画像内の)極小値かを判定
        def isLocalMin(target, rng):
            r = np.array(rng)
            r[1][1][1] = 101 # 自分自身はminから除外する
            min_value = np.min(r)
            if target < min_value:
                return True
            else:
                return False

        for oi, octave in enumerate(dog_space):
            extrema = OrderedDict()
            scales = list(octave.keys())
            for k in range(1,len(scales)-1):
                extrema[scales[k]] = []
                images = np.array([octave[scales[k-1]], octave[scales[k]], octave[scales[k+1]]])
                h,w = images[0].shape
                for i in range(1,h-1):
                    for j in range(1,w-1):
                        # ターゲットの値を取得
                        v = images[1][i][j]  # ターゲットの値
                        # threshold以下の場合はスキップ
                        if abs(v) < threshold: continue
                        # k-1からk+1, i-1からi+1, j-1からj+1の9個のセルを取得
                        rng = images[:,i-1:i+2,j-1:j+2]
                        # 極大を判定
                        if isLocalMax(v, rng):
                            extrema[scales[k]].append((i, j))
                        # 極小を判定
                        if isLocalMin(v, rng):
                            extrema[scales[k]].append((i, j))
            self.append(extrema)

if __name__ == '__main__':
    test_image = Image.open('img/Lena.png').convert('L')
    #test_image = test_images.rectangle(50, 50, 50, 50)
    #test_image = test_images.ellipse(50, 50, 50, 100)
    test_image = np.array(test_image, dtype=np.float) / 255

    scale_space = ScaleSpace(test_image)
    scale_space.create()

    dog_space = DoGSpace(scale_space)
    dog_space.create()

    extrema_space1 = ExtremaSpace(dog_space)
    extrema_space1.find()
    extrema_space1.localize()

    extrema_space2 = ExtremaSpace(dog_space)
    extrema_space2.findByNoguchi(threshold=0.0)
    extrema_space2.find()

    extrema_space = extrema_space2

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

    plt.imshow(test_image, cmap='Greys_r')
    fig = plt.gcf()
    for n, octave in enumerate(extrema_space):
        r = 1
        for scale in octave.keys():
            for p in octave[scale]:
                fig.gca().add_artist(plt.Circle((p[1] * np.power(2, n), p[0] * np.power(2, n)), r, color='r', fill=False))
    plt.show()
