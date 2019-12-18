import numpy as np


def generate_vlp_input_file(infile, outfile):
    content = []
    label2ans = lambda r: 'yes' if r == 'ITSELF' else 'no'
    synset2n = dict()
    with open(infile, 'r') as f:
        i = 0
        for line in f.readlines():
            synset = line.split('\t')[1]
            if synset not in synset2n:
                synset2n[synset] = 0
            if synset2n[synset] < 5:
                i += 1
                img_dict = dict()
                img_dict['synset'] = synset
                img_dict['offset'] = synset2n[synset]
                img_dict['question_id'] = i
                img_dict['image_name'] = line.split('\t')[2]
                img_dict['feature_path'] = '_feats.pkl'
                img_dict['question_str'] = '{}: {}'.format(' '.join(line.split('\t')[0].split('_')), line.split('\t')[4])
                img_dict['has_answer'] = True
                img_dict['answers'] = [label2ans(line.split('\t')[3])]
                synset2n[synset] += 1
                content.append(img_dict)
        print('Instances: ', len(content), ' Concepts: ', len(synset2n))
        np.save(outfile + '_vqa_sensemb.npy', content, allow_pickle=True)


def readable_neighbours(dataset, neigh_file, hr_file):
    synset2lemma = dict()
    synset2gloss = dict()
    with open(dataset, 'r') as f:
        for line in f.readlines():
            synset = line.split('\t')[1]
            if synset not in synset2lemma:
                synset2lemma[synset] = ''.join(line.split('\t')[0].split('_'))
                synset2gloss[synset] = line.split('\t')[4]
    with open(neigh_file, 'r') as f, open(hr_file, 'w+') as w:
        for line in f.readlines():
            synsets = line.split()
            for i, synset in enumerate(synsets):
                offset = '' if i == 0 else '\t'
                w.write('{}{}\t{}\t{}\n'.format(offset, synset, synset2lemma[synset], synset2gloss[synset]))


def main():
    # generate_vlp_input_file('../Image2Synset/DATA/true_examples_15.txt', 'babelpic_gold')
    readable_neighbours('../Image2Synset/DATA/true_examples_15.txt', 'neighbours_cat.tsv', 'neighbours_cat_hr.tsv')


if __name__ == '__main__':
    main()
