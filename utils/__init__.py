from fractions import Fraction
from pathlib import Path
from json import JSONEncoder
from .utils import *


def rmtree(path: Path):
    if path.is_file():
        path.unlink()
    else:
        for ch in path.iterdir():
            rmtree(ch)
        path.rmdir()


def safe_save(fpath, data, save_fun, rewrite=False, error_msg='File {fpath} exists! To rewite it use `--rewrite` flag', **kwargs):
    if not fpath.is_file() or rewrite:
        save_fun(str(fpath), data, **kwargs)
    else:
        raise FileExistsError(error_msg.format(fpath=fpath))


class FractionJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Fraction):
            return {'Fraction': [o.numerator, o.denominator]}
        else:
            return o.__dict__


def fraction_from_json(json_object):
    if 'Fraction' in json_object:
        return Fraction(*json_object['Fraction'])
    return json_object


