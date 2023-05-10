import os

# resDir = './data2'
# labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def getFlist(path, labels):
    root_dirs = []

    for root, dirs, files in os.walk(path):
        print('root_dir:', root)
        print('sub_dirs:', dirs)
        # print('files:', files)
        root_dirs.append(root)
        print('root_dirs:', root_dirs[1:])
    # root_dirs = root_dirs[1:]

    root_dirs = root_dirs[0]
    dir = []
    for i in labels:
        dir.append(root_dirs + '/' + i)
    return dir

def getChildList(root_dirs, resDir):
    j = 0
    f = open(resDir + '/cow_jpg.lst', 'w')

    for path in root_dirs:
        for root, dirs, files in os.walk(path):
            print('child_root_dir:', root)
            print('child_sub_dirs:', dirs)
            # print('child_files:', files)
            for file in files:
                f.write('%s/%s %i\n'%(root,file,j))

        j = j+1
    f.close()
