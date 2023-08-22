import os.path

import numpy as np
from cv2 import dct
from matplotlib import pyplot as plt
from numpy.linalg import svd

from text_watermark.functions import text_core_function, random_strategy1, read_img


def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]  # 1. 初始化中心点
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold  # 2. 检查所有点与这k个点之间的距离，每个点归类到最近的中心
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]  # 3. 重新找中心点
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol:  # 4. 停止条件
            threshold = (center[0] + center[1]) / 2
            break

    is_class01 = inputs > threshold
    return is_class01




class extractor(text_core_function):
    def __init__(self, encoding='gbk', mode='str'):
        super().__init__(encoding=encoding, mode=mode)

        self.sss = []

    def extract_form_file(self, wm_shape=240, filename=None):
        assert os.path.exists(filename), '文件不存在'
        self.ex_img = read_img(filename)

        wm_avg = self.extract_with_kmeans(img=self.ex_img, wm_shape=wm_shape)
        wm = self.extract_decrypt(wm_avg=wm_avg)

        byte = ''.join(str((i >= 0.5) * 1) for i in wm)
        self.byte_ = byte


        # idx = []
        # for i in range(len(self.sss)):
        #     idx.append(i)
        #
        # plt.scatter(idx, self.sss, color='blue', marker='o', s=0.1)
        # # 设置标题和轴标签
        # plt.title('散点图示例')
        # plt.xlabel('X 值')
        # plt.ylabel('Y 值')
        # # 显示图形
        # plt.show()

        # print(wm.replace('$$', '\n'))
        if self.out:
            wm = bytes.fromhex(hex(int(byte, base=2))[2:]).decode(self.encoding, errors='replace')
            return wm.replace('$$', '\n')
        else:
            wm = bytes.fromhex(hex(int(byte, base=2))[2:])
            return wm

    def one_block_get_wm(self, args):
        block, shuffler = args
        block_dct_shuffled = dct(block).flatten()[shuffler].reshape(self.block_shape)

        u, s, v = svd(block_dct_shuffled)

        self.sss.append(s[0] % self.d1)

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

    def extract_avg(self, wm_block_bit):
        # 对循环嵌入+3个 channel 求平均
        wm_avg = np.zeros(shape=self.wm_size)

        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()

        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()

        # 提取每个分块埋入的 bit：
        wm_block_bit = self.extract_bit_from_img(img=img)

        # 做平均：
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

    def extract_with_kmeans(self, img, wm_shape):
        wm_avg = self.extract(img=img, wm_shape=wm_shape)

        return one_dim_kmeans(wm_avg)

    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg


if __name__ == "__main__":
    a = extractor(encoding='utf-8')
    a.extract_form_file(filename='test.jpg')
