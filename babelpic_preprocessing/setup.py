import json
import numpy as np


def generate_src_file(image_names, src_file):
    data = dict()
    content = []
    for img in image_names:
        img_dict = dict()
        img_dict['split'] = 'test'
        img_dict['filename'] = img
        img_dict['filepath'] = 'test'
        img_dict['imgid'] = img_dict['filename'][:-4]
        content.append(img_dict)
    data['images'] = content
    with open(src_file, 'w+') as file:
        json.dump(data, file)


def generate_vqa_src_file(input_files):
    label2ans = lambda r: 'yes' if r == 'ITSELF' else 'no'
    for file in input_files:
        content = []
        with open(file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                img_dict = dict()
                img_dict['question_id'] = i
                img_dict['image_name'] = line.split('\t')[2]
                img_dict['feature_path'] = '_feats.pkl'
                # img_dict['question_str'] = line.split('\t')[4]
                img_dict['question_str'] = 'Does the image depict {} ({})?'.format(' '.join(line.split('\t')[0].split('_')), line.split('\t')[4])
                img_dict['has_answer'] = True
                img_dict['answers'] = [label2ans(line.split('\t')[3])]
                content.append(img_dict)
            np.save(file + '_vqa_question.npy', content, allow_pickle=True)


def generate_detectron_src_file(babelpic_file, src_file):
    with open(babelpic_file, 'r') as f:
        lines = f.readlines()
        im_names = [line.split()[2] for line in lines]
    with open(src_file, 'w+') as f:
        f.write('\n'.join(im_names))


def get_all_img_names(detectron_src_file):
    with open(detectron_src_file, 'r') as f:
        lines = f.readlines()
        im_names = [line.split()[0] for line in lines]
    return im_names


def main():
    image_names = ['bn:00000246n_67_1.jpg', 'bn:00000800n_30_1.jpg', 'bn:00001035n_41_1.jpg', 'bn:00001082n_47_1.jpg', 'bn:00001092n_9_1.jpg', 'bn:00001139n_8_1.jpg', 'bn:00001737n_35_1.jpg', 'bn:00002096n_180_1.jpg', 'bn:00077545n_1_1.jpg', 'bn:00095211v_1_1.jpg']
    src_file = 'dataset_babelpic.json'
    generate_src_file(image_names, src_file)


if __name__ == '__main__':
    # main()
    # generate_detectron_src_file('/home/agostina/master/thesis/Image2Synset/DATA/babelpic_dataset_15.tsv', 'ids.tsv')
    # generate_src_file(get_all_img_names('missing_ids.tsv'), 'missing_dataset_babelpic.json')
    # files = ['../Image2Synset/DATA/15_train.tsv.tmp', '../Image2Synset/DATA/15_val.tsv.tmp', '../Image2Synset/DATA/15_test.tsv.tmp', '../Image2Synset/DATA/15_test_hard.tsv.tmp']
    # generate_vqa_src_file(files)
    # generate_vqa_src_file(['/home/agostina/eclipse-workspace/wsd/imagenet_dataset.tsv'])
    # generate_vqa_src_file(['glosses_silver_batch_{}_filtered.tsv'.format(i) for i in range(5)])
    generate_vqa_src_file(['glosses_silver_batch_0_filtered_second_half.tsv'.format(0)])
