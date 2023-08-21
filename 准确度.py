import numpy as np
from matplotlib import pyplot as plt
from text_watermark.extract import extractor
from text_watermark.functions import text_core_function


def str_size_in_01(text: str, encoding='gbk'):
    byte = bin(int(text.encode(encoding).hex(), base=16))[2:]
    size_ = (np.array(list(byte)) == '1')
    return size_.size


def get_the_wm(wm, ratio=1):
    a = text_core_function(encoding='gbk', length_ran=True, ratio=ratio)
    size, limit_wm = a.init_emb_func("Bque.jpg", wm)
    a.embed(filename='test1.jpg')
    ab = extractor(encoding='gbk')
    wm = ab.extract_form_file(filename='test1.jpg', wm_shape=size)
    return wm, limit_wm


def levenshtein_distance_dp(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def similarity_score(s1: str, s2: str) -> float:
    ld = levenshtein_distance_dp(s1, s2)
    return 1 - ld / max(len(s1), len(s2))



if __name__ == "__main__":
    with open('版权.txt', 'r', encoding='utf-8') as f:
        wm_content = f.read()
    compression_ratios, similarities = [], []
    for ratio in np.arange(1, 15, 1):
        compression_ratios.append(ratio)
        wm, limit_wm = get_the_wm(wm_content, ratio)
        limit_wm = limit_wm.replace('$$', '\n')
        similarity = similarity_score(wm, limit_wm)
        similarities.append(similarity)
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(compression_ratios, similarities, marker='o', linestyle='-', color='b')
    # 标题和标签
    plt.title("Similarity vs. Compression Ratio")
    plt.xlabel("Compression Ratio")
    plt.ylabel("Similarity")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 显示图形
    plt.show()
