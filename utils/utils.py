from PIL import Image
import json
import os

def json_read(fname, **kwargs):
    with open(fname) as j:
        data = json.load(j, **kwargs)
    return data


def json_save(fname, data, indent_len=4, **kwargs):
    with open(fname, "w") as f:
        s = json.dumps(data, sort_keys=True, ensure_ascii=False,
                       indent=" " * indent_len, **kwargs)
        f.write(s)


def process_wb_from_txt(txt_path):
    with open(txt_path, 'r') as fh:
        txt = [line.rstrip().split() for line in fh]

    txt = [[float(k) for k in row] for row in txt]

    assert len(txt) in [1, 3]

    if len(txt) == 1:
        # wb vector
        txt = txt[0]

    return txt


def process_ids_from_txt(txt_path):
    with open(txt_path, 'r') as fh:
        temp = fh.read().splitlines()
    return temp


def save_txt(p, s):
    with open(p, 'w') as text_file:
        text_file.write(s)


def downscale_jpg(img_path, new_shape, quality_perc=100):
    img = Image.open(img_path)
    if (img.size[0], img.size[1]) != new_shape:
        new_img = img.resize(new_shape, Image.ANTIALIAS)
        new_img.save(img_path[:-len('.jpg')] + '.jpg',
                     'JPEG', quality=quality_perc)


def rename_img(img_path):
    if img_path.lower().endswith('jpeg'):
        os.rename(img_path, img_path[:-len('jpeg')] + 'jpg')
    else:
        os.rename(img_path, img_path[:-len('JPG')] + 'jpg')
