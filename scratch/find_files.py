import os
import collections

source_dirs = [os.path.expanduser(p) for p in ['~/opt/src/nginx']] #, '~/opt/src/mongo']]

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def listdir_rec(dir):
    return [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dir)) for f in fn]

def is_filetype(filetypes, path, filename):
    abs_filename = os.path.join(path, filename)
    basename, ext = os.path.splitext(filename)
    c_exts = ["."+ext for ext in filetypes]

    is_file = os.path.isfile(abs_filename)
    is_c = (ext in c_exts)

    return is_file and is_c

def filetypes_in_dirs(filetypes, dirs):
    return [filename
            for dirname  in flatten([dirs])
            for filename in listdir_rec(dirname)
            if is_filetype(filetypes, dirname, filename)]

def c_files_in_dirs(dirs):
    return filetypes_in_dirs(["c", "cc", "cpp", "c++", "h", "hh", "hpp", "h++"], dirs)
