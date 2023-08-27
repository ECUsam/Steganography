import cv2
import numpy as np
from matplotlib import pyplot as plt

from text_watermark import attack
from text_watermark.extract import extractor
from text_watermark.functions import text_core_function
from text_watermark.recover import recover_crop, estimate_crop_parameters
import os

if not os.path.exists('output'):
    os.mkdir('output')


def bit_minus(row_bit, aft_bit):
    assert len(row_bit) == len(aft_bit), "比特长度不同"
    n_ = 0
    for i in range(len(row_bit)):
        if row_bit[i] != aft_bit[i]:
            n_ += 1
    return n_


def jpeg_compress(input_filename, output_filename, quality=95):
    img = cv2.imread(input_filename)
    if img is None:
        raise ValueError(f"Failed to load image {input_filename}")
    cv2.imwrite(output_filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


a = text_core_function(encoding='gbk', length_ran=True)
size = a.init_emb_func("Bque.jpg", "深圳杯数学建模挑战赛")[0]
a.embed(filename='output/test_attack.jpg')
byte = a.byte
ori_img_shape = cv2.imread('Bque.jpg').shape[:2]
h, w = ori_img_shape
ab = extractor(encoding='gbk')

ratios = []
i_s = []
byte_minus = []

loc_r = ((0, 0), (1, 1))
x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])

for i in np.arange(10, 110, 5):

    attack.shelter_att(input_filename='output/test_attack.jpg', output_file_name='output/shelter.png', ratio=0.1, n=i)
    wm_extract = ab.extract_form_file(filename='output/shelter.png', wm_shape=size)
    byte_aft = ab.byte_
    minus = bit_minus(byte, byte_aft)
    print("压缩攻击。提取结果：", wm_extract, "比特差：", minus)
    ratio = (len(byte) - minus) / len(byte)
    i_s.append(i)
    ratios.append(ratio)
    byte_minus.append(minus)

plt.figure(figsize=(10, 6))
plt.plot(i_s, ratios, marker='o', linestyle='-', color='b')
# 标题和标签
plt.title("accuracy vs. Shelter Times")
plt.xlabel("Shelter Times")
plt.ylabel("accuracy")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# 在每个点上标注数值
for i, ratio, minus in zip(i_s, ratios, byte_minus):
    plt.text(i, ratio, f'{minus}', ha='center', va='bottom')  # 可以调整 ha 和 va 参数来控制文本位置


plt.tight_layout()

# 显示图形
plt.show()