"""
Demo raw processing pipeline and pipeline executor.
"""

import numpy as np
from raw_prc_pipeline.pipeline_utils import *


class RawProcessingPipelineDemo:
    """
    Demonstration pipeline of raw image processing.

    This pipeline is a baseline pipeline to process raw image.
    The public methods of this class are successive steps of raw image processing pipeline.
    The declaration order of the public methods must correspond to the order in which these methods (steps) are supposed to be called when processing raw image.

    It is assumed that each public method has 2 parameters:
    raw_img : ndarray
        Array with images data.
    img_meta : Dict
        Some metadata of image.
    
    Also each such public method must return an image (ndarray) as the result of processing.
    """
    def __init__(self, illumination_estimation='', denoise_flg=True, tone_mapping='Flash', out_landscape_width=None, out_landscape_height=None):
        """
        RawProcessingPipelineDemo __init__ method.

        Parameters
        ----------
        illumination_estimation : str, optional
            Options for illumination estimation algorithms: '', 'gw', 'wp', 'sog', 'iwp', by default ''.
        denoise_flg : bool, optional
            Denoising flag, by default True.
            If True, resulted images will be denoised with some predefined parameters.
        tone_mapping : str, optional
            Options for tone mapping methods, defined in function `apply_tone_map` from `pipeline_utils` module.
            By default 'Flash'.
        out_landscape_width : int, optional
            The width of output image (when orientation is landscape). If None, the image resize will not be performed.
            By default None.
        out_landscape_height : int, optional
            The height of output image (when orientation is landscape). If None, the image resize will not be performed.
            By default None.
        """
        self.params = locals()
        del self.params['self']

    # Linearization not handled.
    def linearize_raw(self, raw_img, img_meta):
        return raw_img

    def normalize(self, linearized_raw, img_meta):
        return normalize(linearized_raw, img_meta['black_level'], img_meta['white_level'])

    def demosaic(self, normalized, img_meta):
        return simple_demosaic(normalized, img_meta['cfa_pattern'])

    def denoise(self, demosaic, img_meta):
        if not self.params['denoise_flg']:
            return demosaic
        return denoise_image(demosaic)

    def white_balance(self, demosaic, img_meta):
        if self.params['illumination_estimation'] == '':
            wb_params = img_meta['as_shot_neutral']
        else:
            wb_params = illumination_parameters_estimation(
                demosaic, self.params['illumination_estimation'])

        white_balanced = white_balance(demosaic, wb_params)
        return white_balanced

    def xyz_transform(self, white_balanced, img_meta):
        return apply_color_space_transform(white_balanced, img_meta['color_matrix_1'], img_meta['color_matrix_2'])

    def srgb_transform(self, xyz, img_meta):
        return transform_xyz_to_srgb(xyz)

    def tone_mapping(self, srgb, img_meta):
        if self.params['tone_mapping'] is None:
            return apply_tone_map(srgb, 'Base')
        return apply_tone_map(srgb, self.params['tone_mapping'])

    def gamma_correct(self, srgb, img_meta):
        return apply_gamma(srgb)

    def autocontrast(self, srgb, img_meta):
        # return autocontrast(srgb)
        return autocontrast_using_pil(srgb)

    def to_uint8(self, srgb, img_meta):
        return (srgb*255).astype(np.uint8)
    
    def resize(self, img, img_meta):
        if self.params['out_landscape_width'] is None or self.params['out_landscape_height'] is None:
            return img
        return resize_using_pil(img, self.params['out_landscape_width'], self.params['out_landscape_height'])
    
    def fix_orientation(self, img, img_meta):
        return fix_orientation(img, img_meta['orientation'])


class PipelineExecutor:
    """
    Pipeline executor class.

    This class can be used to successively execute the steps of some image pipeline class (for example `RawProcessingPipelineDemo`).
    The declaration order of the public methods of pipeline class must correspond to the order in which these methods (steps) are supposed to be called when processing image.

    It is assumed that each public method of the pipeline class has 2 parameters:
    raw_img : ndarray
        Array with images data.
    img_meta : Dict
        Some meta data of image.
    
    Also each such public method must return an image (ndarray) as the result of processing.
    """
    def __init__(self, img, img_meta, pipeline_obj, first_stage=None, last_stage=None):
        """
        PipelineExecutor __init__ method.

        Parameters
        ----------
        img : ndarray
            Image that should be processed by pipeline.
        img_meta : Dict
            Some image metadata.
        pipeline_obj : pipeline object
            Some pipeline object such as RawProcessingPipelineDemo.
        first_stage : str, optional
            The name of first public method of pipeline object that should be called by PipelineExecutor.
            If None, the first public method from defined in pipeline object will be considered as `first_stage` method.
            By default None.
        last_stage : str, optional
            The name of last public method of pipeline object that should be called by PipelineExecutor.
            If None, the last public method from defined in pipeline object will be considered as `last_stage` method.
            By default None.
        """
        self.pipeline_obj = pipeline_obj
        self.stages_dict = self._init_stages()
        self.stages_names, self.stages = list(
            self.stages_dict.keys()), list(self.stages_dict.values())

        if first_stage is None:
            self.next_stage_indx = 0
        else:
            assert first_stage in self.stages_names, f"Invalid first_stage={first_stage}. Try use the following stages: {self.stages_names}"
            self.next_stage_indx = self.stages_names.index(first_stage)

        if last_stage is None:
            self.last_stage_indx = len(self.stages_names) - 1
        else:
            assert last_stage in self.stages_names, f"Invalid last_stage={last_stage}. Try use the following stages: {self.stages_names}"
            self.last_stage_indx = self.stages_names.index(last_stage)
            if self.next_stage_indx > self.last_stage_indx:
                print(f'Warning: the specified first_stage={first_stage} follows the specified last_stage={last_stage}, so using __call__ no image processing will be done.')

        self.current_image = img
        self.img_meta = img_meta

    def _init_stages(self):
        stages = {func: getattr(self.pipeline_obj, func) for func in self.pipeline_obj.__class__.__dict__ if callable(
            getattr(self.pipeline_obj, func)) and not func.startswith("_")}
        return stages

    @property
    def next_stage(self):
        if self.next_stage_indx < len(self.stages):
            return self.stages_names[self.next_stage_indx]
        else:
            return None

    @property
    def last_stage(self):
        return self.stages_names[self.last_stage_indx]

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_stage_indx < len(self.stages):
            stage_func = self.stages[self.next_stage_indx]
            self.current_image = stage_func(self.current_image, self.img_meta)
            self.next_stage_indx += 1
            return self.current_image
        else:
            raise StopIteration

    def __call__(self):
        """
        PipelineExecutor __call__ method.

        This method will sequentially execute the methods defined in the pipeline object from the `first_stage` to the `last_stage` inclusive.

        Returns
        -------
        ndarray
            Resulted processed raw image.
        """
        for current_image in self:
            if self.next_stage_indx > self.last_stage_indx:
                return current_image
        return self.current_image
