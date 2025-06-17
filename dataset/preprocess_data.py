import codecs
import os
import pickle
import re
import numpy as np
from tqdm import tqdm
from os.path import join
from gensim.models import word2vec
from params import set_params
from dataset.dump_graph import build_graph
from dataset.load_data import load_json
from dataset.save_results import dump_json, check_mkdir
from character.match_name import match_name

_, args = set_params()

puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
             'the', 'by', 'we', 'be', 'is', 'are', 'can']
stopwords_extend = ['university', 'universities', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing','journal', 'science',
                    'sciences', 'international', 'key', 'research', 'engineering','academy', 'state', 'center',
                    'xuebao', 'conference', 'proceedings', 'technology', 'jishu', 'ieee','acta', 'applied',
                    'letters', 'society', 'communications', 'daxue', 'sinica', 'yu', 'gongcheng','usa','xi',
                    'guangzhou','tianjing','pr','wuhan','chengdu','lanzhou','sichuan','dalian']
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                   'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                   'p', 'results','people', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'ministry'
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system', 'sci', 'affiliated'
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                   'time', 'zhejiang', 'used', 'data', 'these', 'chemistry', 'chemical', 'physics', 'medical',
                   'hospital', 'national', 'information', 'beijing', 'lab', 'education','edu', 'ltd', 'co', ]


def save_pickle(data, *paths):
    if len(paths) > 1:
        path = os.path.join(*paths)
    else:
        path = paths[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pubinfo(mode):
    """
    Read pubs' meta-information.
    """
    base = join(args.save_path, "src")

    if mode == 'train':
        pubs = load_json(join(base, "train", "train_pub.json"))
    elif mode == 'valid':
        pubs = load_json(join(base, "valid", "sna_valid_pub.json"))
    elif mode == 'test':
        pubs = load_json(join(base, 'test', 'sna_test_pub.json'))
    else:
        raise ValueError('choose right mode')

    return pubs


def read_raw_pubs(mode):
    """
    Read raw pubs.
    """
    base = join(args.save_path, "src")

    if mode == 'train':
        raw_pubs = load_json(join(base, "train", "train_author.json"))
    elif mode == 'valid':
        raw_pubs = load_json(join(base, "valid", "sna_valid_raw.json"))
    elif mode == 'test':
        raw_pubs = load_json(join(base, "test", "sna_test_raw.json"))
    else:
        raise ValueError('choose right mode')

    return raw_pubs


def unify_name_order(name):
    """
    unifying different orders of name.
    Args:
        name
    Returns:
        name and reversed name
    """
    token = name.split("_")
    name = token[0] + token[1]
    name_reverse = token[1] + token[0]
    if len(token) > 2:
        name = ''.join(token[:])
        name_reverse = token[-1] + ''.join(token[:-1])

    return name, name_reverse

def dump_plain_texts_to_file(raw_data_root, processed_data_root):
    """
    Dump raw publication data to files.
    Plain texts consist of all paper attributes and the authors' names and organizations (except year).
    """
    train_pubs_dict = load_json(os.path.join(raw_data_root, 'train', 'train_pub.json'))
    valid_pubs_dict = load_json(os.path.join(raw_data_root, 'valid', 'sna_valid_pub.json'))

    pubs_dict = {}
    pubs_dict.update(train_pubs_dict)
    pubs_dict.update(valid_pubs_dict)

    try:
        test_pubs_dict = load_json(os.path.join(raw_data_root, 'test', 'sna_test_pub.json'))
        pubs_dict.update(test_pubs_dict)
    except:
        pass

    texts_dir = os.path.join(processed_data_root, 'extract_text')
    os.makedirs(texts_dir, exist_ok=True)
    wf = codecs.open(os.path.join(texts_dir, 'plain_text.txt'), 'w', encoding='utf-8')

    for i, pid in enumerate(tqdm(pubs_dict)):
        paper_features = []
        pub = pubs_dict[pid]

        # Save title
        title = pub["title"]
        pstr = title.strip()
        pstr = pstr.lower()
        pstr = re.sub(puncs, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        title_features = pstr

        # Save keywords
        keywd_features = ""
        word_list = []
        if "keywords" in pub:
            for word in pub["keywords"]:
                word_list.append(word)
            pstr = " ".join(word_list)
            keywd_features = pstr

        org_list = []
        for author in pub["authors"]:
            # Save org (every author's organization)
            if "org" in author:
                org = author["org"]
                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                if pstr:
                    org_list.append(pstr)

        pstr = " ".join(org_list)
        org_features = pstr

        # Save venue
        venue_features = ''
        if "venue" in pub and type(pub["venue"]) is str:
            venue = pub["venue"]
            pstr = venue.strip()
            pstr = pstr.lower()
            pstr = re.sub(puncs, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            venue_features = pstr

        # Save abstract
        abstract_features = ''
        if "abstract" in pub and type(pub["abstract"]) is str:
            abstract = pub["abstract"]
            pstr = abstract.strip()
            pstr = pstr.lower()
            pstr = re.sub(puncs, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            pstr = pstr.replace('\n', '')
            abstract_features = pstr

        paper_features.append(org_features + ' ' + title_features + ' ' + keywd_features + ' ' + venue_features + ' ' + abstract_features + ' ')
        wf.write(' '.join(paper_features) + '\n')

    print(f'All paper texts extracted.')
    wf.close()


def train_w2v_model(processed_data_root):
    texts_dir = join(processed_data_root, 'extract_text')
    sentences = word2vec.Text8Corpus(join(texts_dir, 'plain_text.txt'))

    model_path = join(processed_data_root, 'w2v_model')
    os.makedirs(model_path, exist_ok=True)
    model = word2vec.Word2Vec(sentences, size=100, negative=5, min_count=5, window=5)
    model.save(join(model_path, 'tvt.model'))
    print(f'Finish word2vec training.')


def dump_paper_emb(processed_data_root, model_name):
    """
    dump paper's [title, org, keywords] average word-embedding as semantic feature.
    """
    model_path = join(processed_data_root, 'w2v_model')
    w2v_model = word2vec.Word2Vec.load(join(model_path, f'{model_name}.model'))

    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        for n, name in enumerate(tqdm(raw_pubs)):
            name_pubs = load_json(join(processed_data_root, 'names_pub', mode, name + '.json'))
            text_feature_path = join(processed_data_root, 'paper_emb', name)
            os.makedirs(text_feature_path, exist_ok=True)

            ori_name = name
            name, name_reverse = unify_name_order(name)

            authorname_dict = {}
            ptext_emb = {}
            tcp = set()

            for i, pid in enumerate(name_pubs):
                pub = name_pubs[pid]
                # save authors
                org = ""
                find_author = False
                for author in pub["authors"]:
                    authorname = ''.join(filter(str.isalpha, author['name'])).lower()
                    taken = authorname.split(" ")
                    if len(taken) == 2:
                        authorname = taken[0] + taken[1]
                        authorname_reverse = taken[1] + taken[0]

                        if authorname not in authorname_dict:
                            if authorname_reverse not in authorname_dict:
                                authorname_dict[authorname] = 1
                            else:
                                authorname = authorname_reverse
                    else:
                        authorname = authorname.replace(" ", "")

                    if authorname != name and authorname != name_reverse:
                        pass
                    else:
                        if "org" in author:
                            org = author["org"]
                            find_author = True
                if not find_author:
                    for author in pub['authors']:
                        if match_name(author['name'], ori_name):
                            if "org" in author:
                                org = author['org']
                                break

                pstr = ""
                keyword = ""
                if "keywords" in pub:
                    for word in pub["keywords"]:
                        keyword = keyword + word + " "

                pstr = pub["title"] + " " + keyword + " " + org
                pstr = pstr.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 2]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = [word for word in pstr if word not in stopwords_check]

                words_vec = []
                for word in pstr:
                    if word in w2v_model:
                        words_vec.append(w2v_model[word])
                if len(words_vec) < 1:
                    words_vec.append(np.zeros(100))
                    tcp.add(i)

                ptext_emb[pid] = np.mean(words_vec, 0)

            save_pickle(ptext_emb, join(text_feature_path, 'ptext_emb.pkl'))
            save_pickle(tcp, join(text_feature_path, 'tcp.pkl'))

    print("Finishing dump all paper embd into files.")


def dump_name_pubs():
    """
    Split publications informations by {name} and dump files as {name}.json

    """
    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        pub_info = read_pubinfo(mode)
        file_path = join(args.save_path, 'names_pub', mode)
        if not os.path.exists(file_path):
            check_mkdir(file_path)
        for name in tqdm(raw_pubs):
            name_pubs_raw = {}
            if mode != "train":
                for i, pid in enumerate(raw_pubs[name]):
                    name_pubs_raw[pid] = pub_info[pid]
            else:
                pids = []
                for aid in raw_pubs[name]:
                    pids.extend(raw_pubs[name][aid])
                for pid in pids:
                    name_pubs_raw[pid] = pub_info[pid]

            dump_json(name_pubs_raw, join(file_path, name + '.json'), indent=4)

    print("Finishing dump pubs according to names.")


def dump_features_relations_to_file():
    """
    Generate paper features and relations by raw publication data and dump to files.
    Paper features consist of title, org, keywords. Paper relations consist of author_name, org, venue.
    """
    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        for n, name in tqdm(enumerate(raw_pubs)):

            file_path = join(args.save_path, 'relations', mode, name)
            check_mkdir(file_path)
            coa_file = open(join(file_path, 'paper_author.txt'), 'w', encoding='utf-8')
            cov_file = open(join(file_path, 'paper_venue.txt'), 'w', encoding='utf-8')
            cot_file = open(join(file_path, 'paper_title.txt'), 'w', encoding='utf-8')
            coo_file = open(join(file_path, 'paper_org.txt'), 'w', encoding='utf-8')

            authorname_dict = {}  # maintain a author-name-dict
            pubs_dict = load_json(join(args.save_path, 'names_pub', mode, name + '.json'))

            ori_name = name
            name, name_reverse = unify_name_order(name)

            for i, pid in enumerate(pubs_dict):
                pub = pubs_dict[pid]

                # Save title (relations)
                title = pub["title"]
                pstr = title.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = [word for word in pstr if word not in stopwords_check]
                for word in pstr:
                    cot_file.write(pid + '\t' + word + '\n')

                # Save keywords
                word_list = []
                if "keywords" in pub:
                    for word in pub["keywords"]:
                        word_list.append(word)
                    pstr = " ".join(word_list)
                    pstr = re.sub(' +', ' ', pstr)
                keyword = pstr

                # Save org (relations)
                org = ""
                find_author = False
                for author in pub["authors"]:
                    authorname = ''.join(filter(str.isalpha, author['name'])).lower()

                    token = authorname.split(" ")
                    if len(token) == 2:
                        authorname = token[0] + token[1]
                        authorname_reverse = token[1] + token[0]
                        if authorname not in authorname_dict:
                            if authorname_reverse not in authorname_dict:
                                authorname_dict[authorname] = 1
                            else:
                                authorname = authorname_reverse
                    else:
                        authorname = authorname.replace(" ", "")

                    if authorname != name and authorname != name_reverse:
                        coa_file.write(pid + '\t' + authorname + '\n')  # current name is a name of co-author
                    else:
                        if "org" in author:
                            org = author["org"]  # current name is a name for disambiguating
                            find_author = True

                if not find_author:
                    for author in pub['authors']:
                        if match_name(author['name'], ori_name):
                            if "org" in author:
                                org = author['org']
                                break

                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = [word for word in pstr if word not in stopwords_check]
                pstr = set(pstr)
                for word in pstr:
                    coo_file.write(pid + '\t' + word + '\n')

                # Save venue (relations)
                if pub["venue"]:
                    pstr = pub["venue"].strip()
                    pstr = pstr.lower()
                    pstr = re.sub(puncs, ' ', pstr)
                    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                    pstr = pstr.split(' ')
                    pstr = [word for word in pstr if len(word) > 1]
                    pstr = [word for word in pstr if word not in stopwords]
                    pstr = [word for word in pstr if word not in stopwords_extend]
                    pstr = [word for word in pstr if word not in stopwords_check]
                    for word in pstr:
                        cov_file.write(pid + '\t' + word + '\n')
                    if len(pstr) == 0:
                        cov_file.write(pid + '\t' + 'null' + '\n')

            coa_file.close()
            cov_file.close()
            cot_file.close()
            coo_file.close()
        print(f'Finish {mode} data extracted.')
    print(f'All paper features extracted.')


def preprocess_data():
    raw_data_root = join(args.save_path, 'src')
    processed_data_root = args.save_path
    dump_plain_texts_to_file(raw_data_root, processed_data_root)
    dump_name_pubs()
    train_w2v_model(processed_data_root)
    dump_paper_emb(processed_data_root, model_name='tvt')
    dump_features_relations_to_file()
    build_graph()


if __name__ == "__main__":
    """
    some pre-processing
    """
    preprocess_data()
