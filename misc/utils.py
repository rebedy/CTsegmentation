import os


def pause():
    input("\n\n>>> DY: Press the <Enter> key to continue... \n")
    return None


def print_1by1(target_list):
    for i, row in enumerate(target_list):
        print(row)


def mkdir_ifnot(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError as e:
        print(e.errno)
        print(">>> DY: Failed to create directory, %s" % path)
        raise
    return path


def makedirs_ifnot(path):
    try:
        if not os.path.exists(path):
            os.makedirs(os.path.join(path))
        else:
            pass
            # print(path, " is already exist.\n")
    except OSError as e:
        print(e.errno)
        print(">>> DY: Failed to create directories, %s" % path)
        raise
    return path


def isfile_error(path):
    if os.path.isfile(path):
        return True
    else:
        print(">>> DY: Couldn't walk through the directory, %s" % path)
        raise OSError


def isfile_pass(path):
    if os.path.isfile(path):
        return True
    else:
        return False
