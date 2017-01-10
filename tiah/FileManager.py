from xml.etree.ElementTree import parse
from os import walk,listdir,makedirs,remove
from os.path import splitext,split,exists
from os import path

def mkdir(bpath,dname):
    """

    create absolute path in string type
    if path is not exist, make directory first then create path.
    :param bpath:  root path
    :param dname: directory name
    :return: string
    """
    path = bpath+'/'+dname
    if not exists(path):
        makedirs(path)
    return str(path)


def rm(dpath):
    flist = listdir(dpath)
    for file in flist:
        remove(dpath+'/'+file)

def read_text(path,fname):
    lines = []
    with open(path+'/'+fname+'.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            lines.append(line.strip())

    return lines


def get_file_list(rpath):
    video_name = []
    video_ext = []
    gt_ext = []
    raw = listdir(rpath)
    for file in raw:
        if splitext(file)[-1] == '.xml':
            gt_ext.append(file)
        else:
            video_ext.append(file)
            video_name.append(splitext(file)[0])
    #assert(len(video_ext) == len(gt_ext)) ,'every video should have ground truth'
    return video_name,video_ext,gt_ext


def parse_xml(path,fname):
    tree = parse(path + "/"+fname)
    root = tree.getroot()
    obj_list = root.getiterator('objectlist')
    count = 0
    gt_list = []
    for subList in obj_list:
        if len(subList) is not 0:
            sub_list = []
            for obj in subList:
                count += 1
                ori = int(obj.findtext('orientation'))
                box = obj[1]
                xc = int(box.get('xc'))
                yc = int(box.get('yc'))
                w = int(box.get('w'))
                h = int(box.get('h'))
                x = xc - (w / 2)
                y = yc - (h / 2)

                sub_list.append((ori, x, y, w, h))

            gt_list.append(sub_list)

    return gt_list


def isExist(bpath, fname):
    return path.exists(bpath + '/' + fname)