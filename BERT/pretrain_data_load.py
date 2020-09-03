"""https://paul-hyun.github.io/bert-01/"""

import random
""" 마스크 생성 """
def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    """
    :param tokens: input tokens
    :param mask_cnt: mask token count = (len(tokens)-3) * 0.15
    :param vocab_list: vocab list => if sentencepiecetokenizer: {vocab}.idx_to_token
    :return: [tokens(masked), mask_idx
    """
    cand_idx = []
    for i, token in enumerate(tokens):
        if token == "[CLS]" or token == ['SEP']:
            continue
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"): # '_' : 단어의 시작을 의미
            cand_idx[-1].append(i)
        else:
            cand_idx.append([i])
    random.shuffle(cand_idx)  # e.g [['▁한국', '어'], ['▁모델', '을'], ['▁공유', '합니다', '.']] idx

    mask_lms = []
    for index_set in cand_idx:
        if len(mask_lms) >= mask_cnt:
            break
        if len(mask_lms) + len(index_set) > mask_cnt:
            continue
        for index in index_set:
            masked_token = None
            if random.random() < 0.8: # 80% replace with [MASK]
                masked_token = "[MASK]"
            else:
                if random.random() < 0.5: # 10% keep origin
                    masked_token = tokens[index]
                else: # 10% random word
                    masked_token = random.choice(vocab_list)
            mask_lms.append({"index": index, "label": tokens[index]})
            tokens[index] = masked_token
    mask_lms = sorted(mask_lms, key=lambda x: x["index"])
    mask_idx = [p["index"] for p in mask_lms]
    mask_label = [p["label"] for p in mask_lms]

    return tokens, mask_idx, mask_label


# 최대 길이 초과하는 토큰 자르기
def trim_tokens(tokens_a, tokens_b, max_seq):
    """token A가 길면 앞에서부터 제거 : del tokens_a[0]
       token B가 길면 뒤에서부터 제거 : pop()
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq:
            break

        if len(tokens_a) > len(tokens_b):
            del tokens_a[0]
        else:
            tokens_b.pop()

