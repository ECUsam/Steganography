from text_watermark.extract import extractor
from text_watermark.functions import text_core_function


if __name__ == "__main__":
    with open('版权.txt', 'r', encoding='utf-8') as f:
        wm = f.read()
    print(wm.__sizeof__())

    a = text_core_function(encoding='utf-8')
    a.init_emb_func("Bque.jpg", wm)
    a.test_info()
    a.embed(filename='test1.jpg')

    ab = extractor(encoding='utf-8')
    ab.extract_form_file('test1.jpg')