# Night Photography Rendering Challenge

This repo contains the source code of [Night Photography Rendering Challenge](https://nightimaging.org/).

<!-- ## Brief Challenge description-->
 
## Installation and requirements

To work with code it is recommended to use **Python 3.7+**.
The required packages are listed in `requirements.txt` and can be installed by calling:

```bash
pip install -r requirements.txt
```

## raw_prc_pipeline

Module `raw_prc_pipeline` contains the source code of various methods and functions that can be used for processing raw images, including:

- Parsing metadata (see `raw_prc_pipeline/exif_*.py` files)
- Demosaicing, white_balancing (Gray World, White Patch, Shades of Gray, [Improved White Patch](https://ieeexplore.ieee.org/document/7025121)), tone mapping and other methods (see `raw_prc_pipeline/pipeline_utils.py`)
- Implemtations of demo classes of raw image processing pipeline and image processing pipeline executor (see `raw_prc_pipeline/pipeline.py`)

## data

Directory `data` conatins example of challenge data:

- PNG file with raw image data (see `data/IMG_2304.png`) and
- Corresponding JSON file (see `data/IMG_2304.json`) with necessary metadata including: `black_level`, `white_level`, `cfa_pattern`, `color_matrix_*` and etc. Metadata was extracted using `raw_prc_pipeline` module.

## demo

Directory `demo` contains demonstration script for processing PNG raw images with JSON metadata using implemented classes and finctions from `raw_prc_pipeline`.

To process PNG images with corresponding metadata from `data` directory call the following command:

```bash
python -m demo.process_pngs -p data -ie gw -tm Flash
```

To see other arguments of the script call `python -m demo.process_pngs -h` from the root directory.

Also the visualizations of different stages of implemented demo raw image processing pipeline can be found in the `demo/process_img.ipynb` file.

Also you can use more reproducible way via Docker:

```bash
sudo docker build -t nightimaging .
sudo docker run --rm -u $(id -u):$(id -g) -v $(pwd)/data:/images nightimaging python -m demo.process_pngs -p /images -ie gw -tm Flash

```
