from find_files import c_files_in_dirs
import csv

# read a whole source tree into a CSV so it can be sent to paragraph-vectors

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
