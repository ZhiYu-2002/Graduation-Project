# import os

# class BatchRename():

#     def rename(self):
#         path = 'F:/data_m2caiSeg/train/groundtruth'
#         filelist = os.listdir(path)
#         total_num = len(filelist)
#         i = 000
#         for item in filelist:
#             if item.endswith('.png'):
#                 src = os.path.join(os.path.abspath(path), item)
#                 dst = os.path.join(os.path.abspath(path), 'frame' + str(i) + '.png')
#                 try:
#                     os.rename(src, dst)
#                     i += 1
#                 except:
#                     continue

# if __name__ == '__main__':
#     demo = BatchRename()
#     demo.rename()

import os

class BatchRename():

    def rename(self):
        path = 'E:/new_workplace_2023_for_NewNet3/tmp'
        filelist = os.listdir(path)
        total_num = len(filelist)
        i = 000
        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(path), item)
                dst = os.path.join(os.path.abspath(path), str(i) + '.png')
                try:
                    os.rename(src, dst)
                    i += 1
                except:
                    continue

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()