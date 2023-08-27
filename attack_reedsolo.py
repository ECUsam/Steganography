import cv2
from reedsolo import RSCodec

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


def encode_string(input_string, encoding='utf-8'):
    rs = RSCodec(10)
    input_bytes = input_string.encode(encoding)
    encoded_bytes = rs.encode(input_bytes)
    return encoded_bytes


def decode_string(encoded_bytes, encoding='utf-8'):
    rs = RSCodec(10)
    decoded_tuple = rs.decode(encoded_bytes)
    decoded_bytes = decoded_tuple[0]
    decoded_string = decoded_bytes.decode(encoding)
    return decoded_string


def jpeg_compress(input_filename, output_filename, quality=95):
    img = cv2.imread(input_filename)
    if img is None:
        raise ValueError(f"Failed to load image {input_filename}")
    cv2.imwrite(output_filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


text = "深圳杯数学建模挑战赛"
encoded_text = encode_string(text, encoding='gbk')
a = text_core_function(encoding='gbk', length_ran=True, mode='byte')
size = a.init_emb_func("Bque.jpg", encoded_text)[0]
a.embed(filename='output/test_attack.jpg')
byte = a.byte
ori_img_shape = cv2.imread('Bque.jpg').shape[:2]
h, w = ori_img_shape
ab = extractor(encoding='gbk', mode='byte')
wm_extract = ab.extract_form_file(filename='output/test_attack.jpg', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print("不攻击的提取结果：", wm_extract, "比特差：", minus)

# %%截屏攻击1 = 裁剪攻击 + 缩放攻击 + 知道攻击参数（之后按照参数还原）

loc_r = ((0.01, 0.01), (1, 0.8))
scale = 0.7

x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])
# print(x1, y1, x2, y2)

# 截屏攻击
attack.cut_att3(input_filename='output/test_attack.jpg', output_file_name='output/attacked1.png',
                loc=(x1, y1, x2, y2), scale=scale)

recover_crop(template_file='output/attacked1.png', output_file_name='output/attacked1_recover.png',
             loc=(x1, y1, x2, y2), image_o_shape=ori_img_shape)

ab = extractor(encoding='gbk', mode='byte')
wm_extract = ab.extract_form_file(filename='output/attacked1_recover.png', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print("截屏攻击，知道攻击参数。提取结果：", wm_extract, "比特差：", minus)

# %% 截屏攻击2 = 剪切攻击 + 缩放攻击 + 不知道攻击参数（因此需要 estimate_crop_parameters 来推测攻击参数）
loc_r = ((0.0, 0.0), (1, 0.8))
scale = 0.7

x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])

# print(f'Crop attack\'s real parameters: x1={x1},y1={y1},x2={x2},y2={y2}')
attack.cut_att3(input_filename='output/test_attack.jpg', output_file_name='output/attacked2.png',
                loc=(x1, y1, x2, y2), scale=scale)

# estimate crop attack parameters:
(x1, y1, x2, y2), image_o_shape, score, scale_infer = estimate_crop_parameters(original_file='output/test_attack.jpg',
                                                                               template_file='output/attacked2.png',
                                                                               scale=(0.5, 2), search_num=200)

# print(f'Crop att estimate parameters: x1={x1},y1={y1},x2={x2},y2={y2}, scale_infer = {scale_infer}. score={score}')

# recover from attack:
recover_crop(template_file='output/attacked2.png', output_file_name='output/attacked2_recover.png',
             loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)

wm_extract = ab.extract_form_file(filename='output/attacked2_recover.png', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print("截屏攻击，不知道攻击参数。提取结果：", wm_extract, "比特差：", minus)

# %%裁剪攻击1 = 裁剪 + 不做缩放 + 知道攻击参数
loc_r = ((0.01, 0.01), (0.98, 0.75))
x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])

attack.cut_att3(input_filename='output/test_attack.jpg', output_file_name='output/random_attacked.png',
                loc=(x1, y1, x2, y2), scale=None)
# recover from attack:
recover_crop(template_file='output/random_attacked.png', output_file_name='output/random_attacked_recover.png',
             loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)

wm_extract = ab.extract_form_file(filename='output/random_attacked_recover.png', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print("裁剪攻击，知道攻击参数。提取结果：", wm_extract, "比特差：", minus)

# %% 裁剪攻击2 = 裁剪 + 不做缩放 + 不知道攻击参数
loc_r = ((0.02, 0.01), (0.98, 0.75))
x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])

attack.cut_att3(input_filename='output/test_attack.jpg', output_file_name='output/random_attacked2.png',
                loc=(x1, y1, x2, y2), scale=None)
# print(f'Cut attack\'s real parameters: x1={x1},y1={y1},x2={x2},y2={y2}')

# estimate crop attack parameters:
(x1, y1, x2, y2), image_o_shape, score, scale_infer = estimate_crop_parameters(original_file='output/test_attack.jpg',
                                                                               template_file='output/random_attacked2.png',
                                                                               scale=(1, 1), search_num=None)

# print(f'Cut attack\'s estimate parameters: x1={x1},y1={y1},x2={x2},y2={y2}. score={score}')

# recover from attack:
recover_crop(template_file='output/random_attacked2.png', output_file_name='output/random_attacked2_recover.png',
             loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)

wm_extract = ab.extract_form_file(filename='output/random_attacked2_recover.png', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print("裁剪攻击，不知道攻击参数。提取结果：", wm_extract, "比特差：", minus)

# %%椒盐攻击
ratio = 0.1
attack.salt_pepper_att(input_filename='output/test_attack.jpg', output_file_name='output/jiaoyan.png', ratio=ratio)
# ratio是椒盐概率

# 提取
wm_extract = ab.extract_form_file(filename='output/jiaoyan.png', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print(f"椒盐攻击ratio={ratio}后的提取结果：", wm_extract, "比特差：", minus)

# %%旋转攻击
angle = 60
attack.rot_att(input_filename='output/test_attack.jpg', output_file_name='output/rot.png', angle=angle)
attack.rot_att(input_filename='output/rot.png', output_file_name='output/rot_recover.png', angle=-angle)

wm_extract = ab.extract_form_file(filename='output/rot_recover.png', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print(f"旋转攻击angle={angle}后的提取结果：", wm_extract, "比特差：", minus)

# %%遮挡攻击
n = 58
attack.shelter_att(input_filename='output/test_attack.jpg', output_file_name='output/shelter.png', ratio=0.1, n=n)
wm_extract = ab.extract_form_file(filename='output/shelter.png', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print(f"遮挡攻击{n}次后的提取结果：", wm_extract, "比特差：", minus)

# %%缩放攻击
attack.resize_att(input_filename='output/test_attack.jpg', output_file_name='output/resize_att.png',
                  out_shape=(600, 900))
attack.resize_att(input_filename='output/resize_att.png', output_file_name='output/resize_att_recover.png',
                  out_shape=ori_img_shape[::-1])
# out_shape 是分辨率，需要颠倒一下
wm_extract = ab.extract_form_file(filename='output/resize_att_recover.png', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
try:
    wm_extract = decode_string(wm_extract, encoding='gbk')
    print("缩放攻击后的提取结果：", wm_extract, "比特差：", minus)
except Exception:
    print("缩放攻击无法解密")

# %%亮度攻击

attack.bright_att(input_filename='output/test_attack.jpg', output_file_name='output/bright_att.png', ratio=0.9)
attack.bright_att(input_filename='output/bright_att.png', output_file_name='output/bright_att_recover.png', ratio=1.1)
wm_extract = ab.extract_form_file(filename='output/bright_att_recover.png', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print("亮度攻击后的提取结果：", wm_extract, "比特差：", minus)

jpeg_compress('output/test_attack.jpg', 'jpeg_com.jpeg', 70)
wm_extract = ab.extract_form_file(filename='jpeg_com.jpeg', wm_shape=size)
byte_aft = ab.byte_
minus = bit_minus(byte, byte_aft)
wm_extract = decode_string(wm_extract, encoding='gbk')
print("压缩攻击。提取结果：", wm_extract, "比特差：", minus)
