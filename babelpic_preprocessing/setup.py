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


def main():
    image_names = ['bn:00000007n_0_1.jpg']
    src_file = 'dataset_babelpic.json'
    generate_src_file(image_names, src_file)


if __name__ == '__main__':
    main()