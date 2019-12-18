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
                synset2n[synset] += 1
                img_dict = dict()
                img_dict['question_id'] = i
                img_dict['image_name'] = line.split('\t')[2]
                img_dict['feature_path'] = '_feats.pkl'
                img_dict['question_str'] = '{}: {}'.format(' '.join(line.split('\t')[0].split('_')), line.split('\t')[4])
                img_dict['has_answer'] = True
                img_dict['answers'] = [label2ans(line.split('\t')[3])]
                content.append(img_dict)
        print('Instances: ', len(content), ' Concepts: ', len(synset2n))
        np.save(outfile + '_vqa_sensemb.npy', content, allow_pickle=True)


def main():
    generate_vlp_input_file('../Image2Synset/DATA/true_examples_15.txt', 'babelpic_gold')


if __name__ == '__main__':
    main()
