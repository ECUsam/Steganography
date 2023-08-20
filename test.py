import numpy as np

from text_watermark.extract import extractor
from text_watermark.functions import text_core_function



def str_size_in_01(text: str , encoding='gbk'):
    byte = bin(int(text.encode(encoding).hex(), base=16))[2:]
    size_ = (np.array(list(byte)) == '1')
    return size_.size

if __name__ == "__main__":
    with open('版权.txt', 'r', encoding='utf-8') as f:
        wm = f.read()
    size = str_size_in_01(wm)
    if size > 379200:
        pass

    a = text_core_function(encoding='gbk', length_ran=True)
    size = a.init_emb_func("Bque.jpg", wm)

    a.embed(filename='test1.jpg')

    ab = extractor(encoding='gbk')
    ab.extract_form_file(filename='test1.jpg', wm_shape=size)
