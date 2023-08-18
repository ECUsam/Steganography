import numpy as np
from cv2 import dct
from numpy.linalg import svd

from text_watermark.functions import text_core_function, random_strategy1


class extractor(text_core_function):
    def __init__(self):
        super().__init__()

    def one_block_get_wm(self, args):
        block, shuffler = args
        block_dct_shuffled = dct(block).flatten()[shuffler].reshape(self.block_shape)

        u, s, v = svd(block_dct_shuffled)
        wm = (s[0] % self.d1 > self.d2 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def extract_bit_from_img(self, img):
        self.read_img_to_arr(img=img)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))
        self.idx_shuffle = random_strategy1(seed=self.password,
                                            size=self.block_num,
                                            block_shape=self.block_shape[0] * self.block_shape[1],  # 16
                                            )

        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.one_block_get_wm,
                                                     [(self.ll_block[channel][self.block_index[i]],
                                                       self.idx_shuffle[i])
                                                      for i in range(self.block_num)])
        return wm_block_bit
