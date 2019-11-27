import json


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


def generate_detectron_src_file(babelpic_file, src_file):
    with open(babelpic_file, 'r') as f:
        lines = f.readlines()
        im_names = [line.split()[2] for line in lines]
    with open(src_file, 'w+') as f:
        f.write('\n'.join(im_names))


def main():
    image_names = ['bn:00000246n_67_1.jpg', 'bn:00000800n_30_1.jpg', 'bn:00001035n_41_1.jpg', 'bn:00001082n_47_1.jpg', 'bn:00001092n_9_1.jpg', 'bn:00001139n_8_1.jpg', 'bn:00001737n_35_1.jpg', 'bn:00002096n_180_1.jpg', 'bn:00077545n_1_1.jpg', 'bn:00095211v_1_1.jpg']
    src_file = 'dataset_babelpic.json'
    generate_src_file(image_names, src_file)


if __name__ == '__main__':
    main()
    # generate_detectron_src_file('/home/agostina/master/thesis/Image2Synset/DATA/babelpic_dataset_15.tsv', 'ids.tsv')