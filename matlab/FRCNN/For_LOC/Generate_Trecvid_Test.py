import os
import os.path as osp

if  __name__ == '__main__':
    ImageSet = './LOC/LOC_Split/tv16.loc.testing.data.txt'
    Save_Name = './dataset/test.list'
    if not os.path.isfile(ImageSet):
        print 'Ground File({}) does not exist'.format(ImageSet)
        sys.exit(1)
    file = open(ImageSet, 'r')
    
    otfile = open(Save_Name, 'w')
    while True:
        line = file.readline()
        if line == '':
            break
        line = line.strip('\n').split(' ')
        assert(len(line) == 2)
        line = line[1]
        assert(line[0:4] == 'shot')
        line = line[4:]
        line = line.split('_')
        assert(len(line) == 2)
        files = os.listdir(osp.join('.','LOC','filtered',line[0],line[1]))
        jpeglist = []
        for f in files:
            if f[-4:] == 'jpeg':
                jpeglist.append(f)
            else:
                print '{}/{} Unsupport Suffix : {}'.format(line[0], line[1], f)

        otfile.write('{}/{}   {}\n'.format(line[0], line[1], len(jpeglist)))
        for f in jpeglist:
            otfile.write('{}\t'.format(f))
        if len(jpeglist) > 0:
            otfile.write('\n')

    file.close()
    otfile.close()

