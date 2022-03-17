import pickle
import re
import json
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm


def create_sp_dataset(lines, same_shape_words, common_mistaken_words, pseudo_similar_words, perspell_words, only_one_noise,  rep_list=['swap','drop','add', 'replace', 'none'], probs=[0.05,0.05,0.05, 0.05, 0.8]):
    all_noisy_lines = []
    all_correct_lines = []
    output = []
    for line in tqdm(lines):
        if len(line.split()) < 1:
            continue
        merged_modified_words, correct_words, all_operators, wrong_input, correct_output = _get_line_representation(line, rep_list, probs,
                                                                     same_shape_words=same_shape_words,
                                                                     common_mistaken_words=common_mistaken_words,
                                                                     pseudo_similar_words=pseudo_similar_words,
                                                                     perspell_words=perspell_words,
                                                                     only_one_noise=only_one_noise)
        all_operators = [list(op) for op in all_operators]
        current_item = {'wrong': wrong_input, 'correct': correct_output, 'noise_operators': all_operators}
        output.append(current_item)


    return output


def create_pair_merged_dataset(lines, same_shape_words, common_mistaken_words, pseudo_similar_words, perspell_words, only_one_noise,  rep_list=['swap','drop','add', 'replace', 'none'], probs=[0.05,0.05,0.05, 0.05, 0.8]):
    all_noisy_lines = []
    all_correct_lines = []
    out_wrongs_lines = []
    out_corrects_lines = []
    output = []
    for line in tqdm(lines):
        if len(line.split()) < 1:
            continue
        merged_modified_words, correct_words, all_operators, wrong_input, correct_output = _get_line_representation(line, rep_list, probs,
                                                                     same_shape_words=same_shape_words,
                                                                     common_mistaken_words=common_mistaken_words,
                                                                     pseudo_similar_words=pseudo_similar_words,
                                                                     perspell_words=perspell_words,
                                                                     only_one_noise=only_one_noise)
        all_operators = [list(op) for op in all_operators]
        correct_listed_shape_tokens = []
        current_token = []
        is_merged = False
        assert len(merged_modified_words) == len(correct_words)
        for i, tok in enumerate(merged_modified_words):
            if 'merge' in all_operators[i]:
                current_token.append(correct_words[i])
                is_merged = True
            else:
                if is_merged:
                    correct_listed_shape_tokens.append(current_token)
                    is_merged = False
                    current_token = []
                correct_listed_shape_tokens.append([correct_words[i]])

        if current_token != []:
            correct_listed_shape_tokens.append(current_token)
        out_wrongs_lines.append(wrong_input)
        out_corrects_lines.append(correct_listed_shape_tokens)
    return out_wrongs_lines, out_corrects_lines


    return output
def noise_on_corpus(correct_output_file, noisy_output_file, lines, same_shape_words, common_mistaken_words, pseudo_similar_words, perspell_words, only_one_noise,  rep_list=['swap','drop','add', 'replace', 'none'], probs=[0.05,0.05,0.05, 0.05, 0.8]):
    all_noisy_lines = []
    all_correct_lines = []
    for line in tqdm(lines):
        noisy_line, correct_line, _, _, _ = _get_line_representation(line,rep_list,probs, same_shape_words=same_shape_words, common_mistaken_words=common_mistaken_words, pseudo_similar_words=pseudo_similar_words, perspell_words=perspell_words, only_one_noise=only_one_noise)
        if len(noisy_line) != len(correct_line):
            print(noisy_line)
            print(correct_line)
            continue
#         assert  len(noisy_line) == len(correct_line)
        if len(noisy_line) <=0:
            continue
        c_line = json.dumps(correct_line) + '\n'
        n_line = json.dumps(noisy_line) + '\n'
        correct_output_file.write(c_line)
        noisy_output_file.write(n_line)
#         all_noisy_lines.append(noisy_line)
#         all_correct_lines.append(correct_line)
    return all_noisy_lines, all_correct_lines



def should_not_noise(word):
    number_patt = re.compile('^[+-]?((\d+(\.\d+)?)|(\.\d+))$')
    return len(word) < 2 or word.lower().islower() or number_patt.match(word)


def _get_line_representation(line, rep_list, probs, is_joint_model=True, same_shape_words=None, common_mistaken_words=None, pseudo_similar_words=None, perspell_words=None, only_one_noise=False):
    perspell_prob = 0.1
    same_shape_prob, common_mistaken_prob, MERG_PROB , MERG_AGAIN_PROB = 0.1, 0.1, 0.005, 0.01
    REPLACE_BY_SIMILAR_CHARS_PROB, ADD_NEIGHBOUR_CHARS_PROB, DROP_COMMON_CHARS_PROB = 0.9, 0.7, 0.5
    BREAK_PROB = 0 #without break
    PSEUDO_SIMILAR_PROB = 0.01
    LAST_CHAR_DROP_PROB, VA_V_DROP_PROB, ALEF_V_Y_DROP_PROB = 0.5, 0.8, 0.8
    SIMILAR_CHAR_OVER_REPEATED_CHAR_ADD_PROB = 0.7
    modified_words = []
    all_words = line.split()
    if_merge = False
    all_operators = []
    for i, word in enumerate(all_words):
        # break if noise applied
        if i != 0 and only_one_noise and all_operators[-1] != ['none']:
            for indx in range(i, len(all_words)):
                modified_words.append(all_words[indx])
                all_operators.append(['none'])
            break
        operators = []
        new_word = word
        all_operators.append(['none'])
        # ignore numbers, english words and one character words
        if should_not_noise(word):
            modified_words.append(word)
            continue

        # perspell data
        if random.random() < perspell_prob and word in perspell_words:
            new_word = perspell_words[word]
            # if len(new_word.split()) == 1:
            all_operators[-1].append('PerSpellData')
            modified_words.append(new_word)
            continue

        # گذار -> گزار levenshtain_distance(wrong, correct) =1 and shape(wrong) = shape(correct)
        if random.random() < same_shape_prob and word in same_shape_words:
            new_word = random.choice(same_shape_words[word])
            if len(new_word.split()) == 1:
                all_operators[-1].append('same_shape_word')
                modified_words.append(new_word)
                continue

        #found by word embedding similarity
        if random.random() < common_mistaken_prob and word in common_mistaken_words:
            new_word = random.choice(common_mistaken_words[word])
            if len(new_word.split()) == 1:
                all_operators[-1].append('common_mistakes_by_embedding')
                modified_words.append(new_word)
                continue

        # levenshtain distance(wrong, correct) = 1 and wrong is a real word
        if random.random() < PSEUDO_SIMILAR_PROB and word in pseudo_similar_words:
            new_word = random.choice(pseudo_similar_words[word])
            if len(new_word.split()) == 1:
                all_operators[-1].append('real2real')
                modified_words.append(new_word)
                continue

        rep_type = np.random.choice(rep_list, 1, p=probs)[0]
        if 'swap' in rep_type:
            new_word = get_swap_word_representation(word)
            all_operators[-1].append('swap')
        elif 'drop' in rep_type and len(word) > 2:
            new_word = get_drop_word_representation(word, DROP_COMMON_CHARS_PROB, LAST_CHAR_DROP_PROB, VA_V_DROP_PROB, ALEF_V_Y_DROP_PROB)
            all_operators[-1].append('drop')
        elif 'add' in rep_type:
            new_word = get_add_word_representation(word, ADD_NEIGHBOUR_CHARS_PROB, SIMILAR_CHAR_OVER_REPEATED_CHAR_ADD_PROB)
            all_operators[-1].append('add')
        elif 'replace' in rep_type:
            new_word = get_replace_word_representation(word, REPLACE_BY_SIMILAR_CHARS_PROB)
            all_operators[-1].append('replace')

        if if_merge:
            if random.random() < MERG_AGAIN_PROB:
                all_operators[-1].append('merge')
                new_word = new_word + '###'
                modified_words.append(new_word)
            else:
                # کلمه دوم نمی‌تونه دیگه خطا داشته باشه
                if_merge = False
                modified_words.append(new_word)
            continue

        using_merge = random.random() < MERG_PROB
        if using_merge:
            all_operators[-1].append('merge')
            new_word = new_word + '###'
            modified_words.append(new_word)
            if_merge = True
            continue

        break_word = random.random() < BREAK_PROB
        if len(word) > 4 and break_word:
            i = random.randint(2,len(new_word)-1)
            new_word = new_word[:i] + ' ' + new_word[i:]
            modified_words.append(new_word)
            all_operators[-1].append('break')
            continue

        # else:
        #     #TODO: give a more ceremonious error...
        #     raise NotImplementedError
        # rep.append(word_rep)
        modified_words.append(new_word)
    
    # return rep, " ".join(modified_words)
    correct_output = []
    current_token = []
    correct_words = line.split()
    correct_wrong_operators = zip(correct_words, modified_words, all_operators)
    for index in range(len(all_operators)):
        all_operators[index].remove('none')
        all_operators[index] = set(all_operators[index])

    wrong_input = ''
    merged_modified_words = [m for m in modified_words]
    for index, (correct_word, mod_word, op) in enumerate(correct_wrong_operators):
        if mod_word.endswith('###'):
            merged = ''
            indexes = []
            for j in range(index, len(modified_words)-1):
                indexes.append(j)
                indexes.append(j+1)
                merged += modified_words[j].replace('###', '')
                merged += modified_words[j+1].replace('###', '')
                if not modified_words[j+1].endswith('###'):
                    wrong_input += merged + ' '
                    break
            for i in indexes:
                merged_modified_words[i] = merged
                all_operators[i].add('merge')
        elif index ==0 or (index>0 and not modified_words[index-1].endswith('###')):
            wrong_input += mod_word + ' '
    correct_output = ' '.join(correct_words)
    return merged_modified_words, correct_words, all_operators, wrong_input, correct_output


def get_merge_word_representation(word1, word2):
    return word1 + '###'


def get_replace_word_representation(word, prob):
    p = random.random()
    if p < prob:
        idx = random.randint(0, len(word)-1)
        ch = get_similar_char(word[idx])
        if idx == len(word) - 1:
            word = word[:idx] + ch
        else:
            word = word[:idx] + ch + word[idx+1:]
    else:
        idx = random.randint(0, len(word)-1)
        ch = _get_random_char()
        if idx == len(word) - 1:
            word = word[:idx] + ch
        else:
            word = word[:idx] + ch + word[idx+1:]
    return word


def get_swap_word_representation(word):
    # dirty case
    if len(word) == 1:
        return word

    idx = random.randint(0, len(word)-2)
    word = word[:idx] + word[idx + 1] + word[idx] + word[idx+2:]
    return word

                           
def get_drop_word_representation(word, prob, last_char_prob, va_v_drop_prob, alef_v_y_drop_prob):
    if_drop = False
    p = random.random()
    # good drop
    if p < prob:
        # drop alef vav y
        most_dropped_chars = ['ا', 'و', 'ی', 'ء']
        if random.random() < alef_v_y_drop_prob and any(c in word for c in most_dropped_chars):
            random.shuffle(most_dropped_chars)
            droped_c = [c for c in most_dropped_chars if c in word][0]
            word = word.replace(droped_c, '')
            if_drop = True

        #drop last ch
        elif random.random() > last_char_prob:
            word = word[:-1]
            if_drop = True
        elif 'وا' in word and random.random() < va_v_drop_prob:
            word = word.replace('وا', 'ا')
            if_drop = True
    #random drop
    if not if_drop:
        idx = random.randint(0, len(word)-2)
        word = word[:idx] + word[idx+1:]
    # return rep, word
    return word


def get_add_word_representation(word, prob, similar_char_over_repeated_char_prob):
    p = random.random()
    if p < prob:
        #add neighberhoud char
        if random.random() < similar_char_over_repeated_char_prob:
            idx = random.randint(0, len(word)-1)
            similar_ch = get_similar_char(word[idx])
            p = random.random()
            if p < 0.5:
                word = word[:idx] + word[idx] + similar_ch + word[idx+1:]
            else:
                word = word[:idx] + similar_ch + word[idx]  + word[idx+1:]
        #repeat char
        else:
            idx = random.randint(0, len(word)-1)
            word = word[:idx] + word[idx] + word[idx:]

    else:
        idx = random.randint(0, len(word)-1)

        random_char = _get_random_char()
        word = word[:idx] + random_char + word[idx:]
        # rep, _ = get_swap_word_representation(word) # don't care about the returned word
        _ = get_swap_word_representation(word) # don't care about the returned word
    return word


def _get_random_char():
    alphabets = "ضصثقفغعهخحجچشسیبلاتنمکگظطزرذدپو"
    alphabets = [i for i in alphabets]
    return np.random.choice(alphabets, 1)[0]

def get_isomorph_chars(ch):
    mapper = {}
    char_synsets = ['رزژ', 'فق', 'کگ', 'خحجچ', 'عغ', 'طظ', 'ذد', 'بیپ', 'تن', 'ثت', 'صض', 'یئ']
    for synset in char_synsets:
        mapper.update({w: list(synset.replace(w, '')) for w in synset})
    if ch not in mapper: return ch
    return np.random.choice(mapper[ch], 1)[0]

def get_Homophone(ch):
    mapper = {}
    char_synsets = ['زضذظ', 'عئاآ', 'طت', 'صسث', 'قغ', 'هح']
    for synset in char_synsets:
        mapper.update({w: list(synset.replace(w, '')) for w in synset})
    if ch not in mapper: return ch
    return np.random.choice(mapper[ch], 1)[0]


def _get_keyboard_neighbor(ch, mod):
    # mods = ['only_row', 'only_up_down', 'diameter']
    ranges = {'only_row': (0, 2), 'only_up_down': (2, 4), 'diameter': (4, 8)}
    # global keyboard_mappings
    # if keyboard_mappings is None or len(keyboard_mappings) != 31:
    keyboard_mappings = defaultdict(lambda: [])
    keyboard = ["ضصثقفغعهخحجچ", "شسیبلاتنمکگ*", "ظطزرذدپو****"]
    row = len(keyboard)
    col = len(keyboard[0])
    dx = [0, 0, -1, 1, 1, -1, -1, 1]
    dy = [-1, 1, 0, 0, 1, -1, 1, -1]
    selected_range = ranges[mod]
    for i in range(row):
        for j in range(col):
            for k in range(selected_range[0], selected_range[1]):
                x_, y_ = i + dx[k], j + dy[k]
                if (x_ >= 0 and x_ < row) and (y_ >= 0 and y_ < col):
                    if keyboard[x_][y_] == '*': continue
                    if keyboard[i][j] == '*': continue

                    keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])

    if ch not in keyboard_mappings: return ch
    return np.random.choice(keyboard_mappings[ch], 1)[0]


def get_similar_char(ch):
    actions = ['key_only_row', 'key_only_up_down', 'key_diameter', 'homophone', 'isomorph']
    probs = [0.3, 0.2, 0.1, 0.2, 0.2]
    act = np.random.choice(actions, 1, p=probs)[0]
    if act == 'key_only_row':
        new_ch = _get_keyboard_neighbor(ch, mod='only_row')
    elif act == 'key_only_up_down':
        new_ch = _get_keyboard_neighbor(ch, mod='only_up_down')
    if act == 'key_diameter':
        new_ch = _get_keyboard_neighbor(ch, mod='diameter')
    if act == 'homophone':
        new_ch = get_Homophone(ch)
    if act == 'isomorph':
        new_ch = get_isomorph_chars(ch)
    return new_ch
