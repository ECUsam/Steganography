from text_watermark.functions import text_core_function

if __name__ == "__main__":
    a = text_core_function(encoding='gbk')
    a.init_emb_func("Bque.jpg", '深圳杯数学建模挑战赛')
    a.test_info()
    a.embed(filename='test.jpg')
    