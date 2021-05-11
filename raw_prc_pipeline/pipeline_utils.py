"""
Camera pipeline utilities.
"""

import os
from fractions import Fraction

import cv2
import numpy as np
import exifread
# from exifread import Ratio
from exifread.utils import Ratio
import rawpy
from scipy.io import loadmat
from raw_prc_pipeline.exif_utils import parse_exif, get_tag_values_from_ifds
from raw_prc_pipeline.fs import perform_storm, perform_flash
from PIL import Image, ImageOps
from skimage.restoration import denoise_bilateral
from skimage.transform import resize as skimage_resize


def get_visible_raw_image(image_path):
    raw_image = rawpy.imread(image_path).raw_image_visible.copy()
    # raw_image = rawpy.imread(image_path).raw_image.copy()
    return raw_image


def get_image_tags(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    return tags


def get_image_ifds(image_path):
    ifds = parse_exif(image_path, verbose=False)
    return ifds


def get_metadata(image_path):
    metadata = {}
    tags = get_image_tags(image_path)
    ifds = get_image_ifds(image_path)
    metadata['linearization_table'] = get_linearization_table(tags, ifds)
    metadata['black_level'] = get_black_level(tags, ifds)
    metadata['white_level'] = get_white_level(tags, ifds)
    metadata['cfa_pattern'] = get_cfa_pattern(tags, ifds)
    metadata['as_shot_neutral'] = get_as_shot_neutral(tags, ifds)
    color_matrix_1, color_matrix_2 = get_color_matrices(tags, ifds)
    metadata['color_matrix_1'] = color_matrix_1
    metadata['color_matrix_2'] = color_matrix_2
    metadata['orientation'] = get_orientation(tags, ifds)
    # isn't used
    metadata['noise_profile'] = get_noise_profile(tags, ifds)
    # ...
    # fall back to default values, if necessary
    if metadata['black_level'] is None:
        metadata['black_level'] = 0
        print("Black level is None; using 0.")
    if metadata['white_level'] is None:
        metadata['white_level'] = 2 ** 16
        print("White level is None; using 2 ** 16.")
    if metadata['cfa_pattern'] is None:
        metadata['cfa_pattern'] = [0, 1, 1, 2]
        print("CFAPattern is None; using [0, 1, 1, 2] (RGGB)")
    if metadata['as_shot_neutral'] is None:
        metadata['as_shot_neutral'] = [1, 1, 1]
        print("AsShotNeutral is None; using [1, 1, 1]")
    if metadata['color_matrix_1'] is None:
        metadata['color_matrix_1'] = [1] * 9
        print("ColorMatrix1 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['color_matrix_2'] is None:
        metadata['color_matrix_2'] = [1] * 9
        print("ColorMatrix2 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['orientation'] is None:
        metadata['orientation'] = 0
        print("Orientation is None; using 0.")
    # ...
    return metadata


def get_linearization_table(tags, ifds):
    possible_keys = ['Image Tag 0xC618', 'Image Tag 50712',
                     'LinearizationTable', 'Image LinearizationTable']
    return get_values(tags, possible_keys)


def get_black_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61A', 'Image Tag 50714',
                     'BlackLevel', 'Image BlackLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Black level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50714, ifds)
    return vals


def get_white_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61D', 'Image Tag 50717',
                     'WhiteLevel', 'Image WhiteLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("White level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50717, ifds)
    return vals


def get_cfa_pattern(tags, ifds):
    possible_keys = ['CFAPattern', 'Image CFAPattern']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("CFAPattern not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(33422, ifds)
    return vals


def get_as_shot_neutral(tags, ifds):
    possible_keys = ['Image Tag 0xC628', 'Image Tag 50728',
                     'AsShotNeutral', 'Image AsShotNeutral']
    return get_values(tags, possible_keys)


def get_color_matrices(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC621', 'Image Tag 50721',
                       'ColorMatrix1', 'Image ColorMatrix1']
    color_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC622', 'Image Tag 50722',
                       'ColorMatrix2', 'Image ColorMatrix2']
    color_matrix_2 = get_values(tags, possible_keys_2)
    #print(f'Color matrix 1:{color_matrix_1}')
    #print(f'Color matrix 2:{color_matrix_2}')
    #print(np.sum(np.abs(np.array(color_matrix_1) - np.array(color_matrix_2))))
    return color_matrix_1, color_matrix_2


def get_orientation(tags, ifds):
    possible_tags = ['Orientation', 'Image Orientation']
    return get_values(tags, possible_tags)


def get_noise_profile(tags, ifds):
    possible_keys = ['Image Tag 0xC761', 'Image Tag 51041',
                     'NoiseProfile', 'Image NoiseProfile']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Noise profile not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(51041, ifds)
    return vals


def get_values(tags, possible_keys):
    values = None
    for key in possible_keys:
        if key in tags.keys():
            values = tags[key].values
    return values


def normalize(raw_image, black_level, white_level):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        if type(black_level[0]) is Fraction:
            black_level = fractions2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]
    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level
    normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    return normalized_image


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def fractions2floats(fractions):
    floats = []
    for fraction in fractions:
        floats.append(float(fraction.numerator) / fraction.denominator)
    return floats


def illumination_parameters_estimation(current_image, illumination_estimation_option):
    ie_method = illumination_estimation_option.lower()
    if ie_method == "gw":
        ie = np.mean(current_image, axis=(0, 1))
        ie /= ie[1]
        return ie
    elif ie_method == "sog":
        sog_p = 4.
        ie = np.mean(current_image**sog_p, axis=(0, 1))**(1/sog_p)
        ie /= ie[1]
        return ie
    elif ie_method == "wp":
        ie = np.max(current_image, axis=(0, 1))
        ie /= ie[1]
        return ie
    elif ie_method == "iwp":
        samples_count = 20
        sample_size = 20
        rows, cols = current_image.shape[:2]
        data = np.reshape(current_image, (rows*cols, 3))
        maxima = np.zeros((samples_count, 3))
        for i in range(samples_count):
            maxima[i, :] = np.max(data[np.random.randint(
                low=0, high=rows*cols, size=(sample_size)), :], axis=0)
        ie = np.mean(maxima, axis=0)
        ie /= ie[1]
        return ie
    else:
        raise ValueError(
            'Bad illumination_estimation_option value! Use the following options: "gw", "wp", "sog", "iwp"')


def white_balance(demosaic_img, as_shot_neutral):
    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)

    as_shot_neutral = np.asarray(as_shot_neutral)
    # transform vector into matrix
    if as_shot_neutral.shape == (3,):
        as_shot_neutral = np.diag(1./as_shot_neutral)

    assert as_shot_neutral.shape == (3, 3)

    white_balanced_image = np.dot(demosaic_img, as_shot_neutral.T)
    white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)

    return white_balanced_image


def simple_demosaic(img, cfa_pattern):
    raw_colors = np.asarray(cfa_pattern).reshape((2, 2))
    demosaiced_image = np.zeros((img.shape[0]//2, img.shape[1]//2, 3))
    for i in range(2):
        for j in range(2):
            ch = raw_colors[i, j]
            if ch == 1:
                demosaiced_image[:, :, ch] += img[i::2, j::2] / 2
            else:
                demosaiced_image[:, :, ch] = img[i::2, j::2]
    return demosaiced_image


def denoise_image(demosaiced_image):
    current_image = denoise_bilateral(
        demosaiced_image, sigma_color=None, sigma_spatial=1., multichannel=True, mode='reflect')
    return current_image


def apply_color_space_transform(demosaiced_image, color_matrix_1, color_matrix_2):
    if isinstance(color_matrix_1[0], Fraction):
        color_matrix_1 = fractions2floats(color_matrix_1)
    if isinstance(color_matrix_2[0], Fraction):
        color_matrix_2 = fractions2floats(color_matrix_2)
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))
    # normalize rows (needed?)
    xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)
    xyz2cam2 = xyz2cam2 / np.sum(xyz2cam1, axis=1, keepdims=True)
    # inverse
    cam2xyz1 = np.linalg.inv(xyz2cam1)
    cam2xyz2 = np.linalg.inv(xyz2cam2)
    # for now, use one matrix  # TODO: interpolate btween both
    # simplified matrix multiplication
    xyz_image = cam2xyz1[np.newaxis, np.newaxis, :, :] * \
        demosaiced_image[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)
    xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image


def transform_xyz_to_srgb(xyz_image):
    # srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
    #                      [0.2126729, 0.7151522, 0.0721750],
    #                      [0.0193339, 0.1191920, 0.9503041]])

    # xyz2srgb = np.linalg.inv(srgb2xyz)

    xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])

    # normalize rows (needed?)
    xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    srgb_image = xyz2srgb[np.newaxis, np.newaxis,
                          :, :] * xyz_image[:, :, np.newaxis, :]
    srgb_image = np.sum(srgb_image, axis=-1)
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def reverse_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW
    rev_orientations = np.array([1, 2, 3, 4, 5, 8, 7, 6])
    return fix_orientation(image, rev_orientations[orientation - 1])


def apply_gamma(x):
    # return x ** (1.0 / 2.2)
    x = x.copy()
    idx = x <= 0.0031308
    x[idx] *= 12.92
    x[idx == False] = (x[idx == False] ** (1.0 / 2.4)) * 1.055 - 0.055
    return x


def apply_tone_map(x, tone_mapping='Base'):
    if tone_mapping == 'Flash':
        return perform_flash(x, perform_gamma_correction=0)/255.
    elif tone_mapping == 'Storm':
        return perform_storm(x, perform_gamma_correction=0)/255.
    elif tone_mapping == 'Drago':
        tonemap = cv2.createTonemapDrago()
        return tonemap.process(x.astype(np.float32))
    elif tone_mapping == 'Mantiuk':
        tonemap = cv2.createTonemapMantiuk()
        return tonemap.process(x.astype(np.float32))
    elif tone_mapping == 'Reinhard':
        tonemap = cv2.createTonemapReinhard()
        return tonemap.process(x.astype(np.float32))
    elif tone_mapping == 'Linear':
        return np.clip(x/np.sort(x.flatten())[-50000], 0, 1)
    elif tone_mapping == 'Base':
        # return 3 * x ** 2 - 2 * x ** 3
        # tone_curve = loadmat('tone_curve.mat')
        tone_curve = loadmat(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'tone_curve.mat'))
        tone_curve = tone_curve['tc']
        x = np.round(x * (len(tone_curve) - 1)).astype(int)
        tone_mapped_image = np.squeeze(tone_curve[x])
        return tone_mapped_image
    else:
        raise ValueError(
            'Bad tone_mapping option value! Use the following options: "Base", "Flash", "Storm", "Linear", "Drago", "Mantiuk", "Reinhard"')


def autocontrast(output_image, cutoff_prcnt=2, preserve_tone=False):
    if preserve_tone:
        min_val, max_val = np.percentile(output_image, [cutoff_prcnt, 100 - cutoff_prcnt])
        output_image = (output_image - min_val)/(max_val - min_val)
    else:
        channels = [None]*3
        for ch in range(3):
            min_val, max_val = np.percentile(output_image[...,ch], [cutoff_prcnt, 100 - cutoff_prcnt])
            channels[ch] = (output_image[...,ch] - min_val)/(max_val - min_val)
        output_image = np.dstack(channels)
    output_image = np.clip(output_image, 0, 1)
    return output_image


def autocontrast_using_pil(img, cutoff=2):
    img_uint8 = np.clip(255*img, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_pil = ImageOps.autocontrast(img_pil, cutoff=cutoff)
    output_image = np.array(img_pil).astype(np.float32) / 255
    return output_image


def raw_rgb_to_cct(rawRgb, xyz2cam1, xyz2cam2):
    """Convert raw-RGB triplet to corresponding correlated color temperature (CCT)"""
    pass
    # pxyz = [.5, 1, .5]
    # loss = 1e10
    # k = 1
    # while loss > 1e-4:
    #     cct = XyzToCct(pxyz)
    #     xyz = RawRgbToXyz(rawRgb, cct, xyz2cam1, xyz2cam2)
    #     loss = norm(xyz - pxyz)
    #     pxyz = xyz
    #     fprintf('k = %d, loss = %f\n', [k, loss])
    #     k = k + 1
    # end
    # temp = cct


def resize_using_skimage(img, width=1296, height=864):
    out_shape = (height, width) + img.shape[2:]
    if img.shape == out_shape:
        return img
    out_img = skimage_resize(img, out_shape, preserve_range=True, anti_aliasing=True)
    out_img = out_img.astype(np.uint8)
    return out_img


def resize_using_pil(img, width=1296, height=864):
    img_pil = Image.fromarray(img)
    out_size = (width, height)
    if img_pil.size == out_size:
        return img
    out_img = img_pil.resize(out_size, Image.ANTIALIAS)
    out_img = np.array(out_img)
    return out_img


def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image