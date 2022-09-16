import os
import json
import pandas as pd



def get_unicode_num(chosung_idx, jungsung_idx, jongsung_idx):
    return ((chosung_idx * 588) + (jungsung_idx * 28) + jongsung_idx) + 44032


def create_name_word():
    chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
                    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                     'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

    jongsung_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
                     'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
                     'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    with open('able_word.txt', 'w', encoding='UTF-8') as f:
        for i in range(len(chosung_list)):
            for j in range(len(jungsung_list)):
                if j != 3:
                    for k in range(len(jongsung_list)):
                        if k not in [2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27]:
                            f.write(chr(get_unicode_num(i, j, k)) + ' ')

    with open('able_word.txt', 'r', encoding='UTF-8') as f:
        able_word = f.readline().split(' ')

    with open('ksx1001.txt', 'r', encoding='UTF-8') as f:
        ksx_word = f.readline().split(' ')

    result = []
    for a in able_word:
        for b in ksx_word:
            if a == b:
                result.append(a)

    result.reverse()
    print(len(result))

    with open('name_word.txt', 'w', encoding='UTF-8') as f:
        while result:
            f.writelines(result.pop() + '\n')

    word_list = []
    with open('name_word.txt', 'r', encoding='UTF-8') as f:
        while True:
            word = f.readline()
            if not word: break
            if word == '\n': continue
            word_list.append(word.rstrip('\n'))

    word_to_index = {word: idx for idx, word in enumerate(word_list)}
    index_to_word = {idx: word for idx, word in enumerate(word_list)}

    with open('index_to_word.txt', 'w', encoding='UTF-8') as f:
        json.dump(index_to_word, f, indent=2, ensure_ascii=False)

    with open('word_to_index.txt', 'w', encoding='UTF-8') as f:
        json.dump(word_to_index, f, indent=2, ensure_ascii=False)


def create_label_file(dataset_path='environment/TextRecognitionDataGenerator/trdg/Training/'):
    with open('word_to_index.txt', 'r', encoding='UTF-8') as f:
        word_to_index = json.load(f)

    path_list = os.listdir(dataset_path)
    label = []
    file_name_list = []
    for file_name in path_list:
        label.append(word_to_index[file_name[0]])
        file_name_list.append(file_name)
    df = pd.DataFrame({'file_name': file_name_list, 'label': label})
    df.to_csv('train_labels.csv', index=False)

def create_custom_dataset(label_path,dataset_path):
    dataset = CustomImageDataset(label_path,dataset_path)
    dataloader = DataLoader(dataset,batch_size=256,shuffle=True)
    return dataset, dataloader

