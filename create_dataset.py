import glob
import json
import pickle
from hazm import Normalizer, WordTokenizer
from utils import create_sp_dataset


def get_raw_text(news_archive_addr):
    lines = []
    for addr in glob.iglob(f'{news_archive_addr}/*.json') :
        data = json.load(open(addr))
        descs = [item['title'] for item in data if 'title' in item and item['title'] is not None]
        lines.extend(descs)
    lines = [normalizer.normalize(' '.join(tokenizer.tokenize(l))) for l in lines]
    lines = [l.replace('_', ' ').replace('-', ' ') for l in lines]
    return lines

if __name__ == '__main__':
    tokenizer = WordTokenizer()
    normalizer = Normalizer()
    SAME_SHAPE_WORDS_ADDR = 'data/similar_shape_words.pkl'
    COMMON_MISTAKEN_WORDS_ADDR = 'data/common_mistaken_words.pkl'
    PSEUDO_SIMILAR_WORDS_ADDR = 'data/reals_mapper_distance1_100f_vocab.pkl'
    PerSpellData_ADDR = 'data/PerSpellData_mapper.pkl'
    VOCAB_ADDR  = 'data/norm_formal_persian_words.json'
    NEWS_ARCHIVE_ADDR = 'data/news_archive'

    # files
    output_json_addr = 'single_noise_sp_dataset_news_titles.json'
    same_shape_words = pickle.load(open(SAME_SHAPE_WORDS_ADDR, 'rb'))
    common_mistaken_words = pickle.load(open(COMMON_MISTAKEN_WORDS_ADDR, 'rb'))
    pseudo_similar_words = pickle.load(open(PSEUDO_SIMILAR_WORDS_ADDR, 'rb'))
    perspell_words = pickle.load(open(PerSpellData_ADDR, 'rb'))
    vocab = json.load(open(VOCAB_ADDR))

    #load raw texts
    lines = get_raw_text(NEWS_ARCHIVE_ADDR)

    #apply noise on correct inputs
    output = create_sp_dataset(lines, same_shape_words, common_mistaken_words, pseudo_similar_words,
                          perspell_words, only_one_noise=True, rep_list=['swap', 'drop', 'add', 'replace', 'none'],
                          probs=[0.05, 0.05, 0.05, 0.05, 0.8])

    #remove non changing
    print(len(output))
    mis_lengths = [item for item in output if not (len(item['wrong'].split()) == len(item['correct'].split()) == len(item['noise_operators'])) ]
    output = [out for out in output if not all(op == [] for op in out['noise_operators'])]
    for item in output:
        item['noise_type'] = [op for op in item['noise_operators'] if op][0]

    #set if_real property
    for item in output:
        try:
            wrong_word = [w for (w,c, op) in zip(item['wrong'].split(), item['correct'].split(), item['noise_operators']) if op][0]
        except :
            item['if_real'] = False
            continue
        if wrong_word in vocab:
            item['if_real'] = True
        else:
            item['if_real'] = False
    print(len(output))
    #save dataset
    with open(output_json_addr, 'w+', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)