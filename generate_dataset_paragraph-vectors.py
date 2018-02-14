# read a whole source tree into a CSV so it can be sent to paragraph-vectors
import os
import csv
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

def is_c_file(path, filename):
    abs_filename = os.path.join(path, filename)
    basename, ext = os.path.splitext(filename)
    c_exts = [".c", ".cc", ".cpp", ".c++", ".h", ".hh", ".hpp", ".h++"]

    is_file = os.path.isfile(abs_filename)
    is_c = (ext in c_exts)

    return is_file and is_c

def c_files_in_dirs(dirs):
    return [filename
            for dirname  in flatten([dirs])
            for filename in listdir_rec(dirname)
            if is_c_file(dirname, filename)]

def main():
    c_files = c_files_in_dirs(source_dirs)

    with open('nginx_source.csv', "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["source","filename"])

        for filename in c_files:
            with open(filename, 'r') as file_obj:
                try:
                    file_content = file_obj.read()
                    if len(file_content) < 131072: # CSV Field Limit
                        csv_writer.writerow([file_content, filename])
                except:
                    None


if __name__ == "__main__":
    main()
