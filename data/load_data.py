import os
import glob
import pandas as pd

from sklearn.model_selection import train_test_split

def get_path(args):
    path = f"F:\\HIPPO\\DATA\\"
    f=[]

    for folder in os.listdir(path):
        f.append(folder)

    train_folder, test_folder,_, _ = train_test_split(f, f, test_size=0.15, random_state=args.seed)

    train_ct, train_hippo = [], []
    test_ct, test_hippo = [], [] 

    w = ['2188579', '9463969', '3939496',
         '1456634', '6678212', '6144646', '9462219',
         '6107464', '6282980', '6290905', '6479636', '10155794',
         '6477005', '3289530']

    for train_f in train_folder:
        if train_f in w:
            continue
        folder_path = f"{path}{train_f}"
        for file in glob.glob(folder_path + "/*.nii"):
            filename = file.split('\\')[4]
            if filename.startswith('r') and filename.endswith('_CT.nii'):
                train_ct.append(file)
            elif filename.startswith('lh+rh'):
                train_hippo.append(file)
            # elif filename.startswith('boundary'):
            #     train_edge.append(file)

    for test_f in test_folder:
        if test_f in w:
            continue
        folder_path = f"{path}{test_f}"
        for file in glob.glob(folder_path + "/*.nii"):
            filename = file.split('\\')[4]
            if filename.startswith('r') and filename.endswith('_CT.nii'):
                test_ct.append(file)
            elif filename.startswith('lh+rh'):
                test_hippo.append(file)
            # elif filename.startswith('edge'):
            #     test_edge.append(file)
                
    print(f"Train CT : {len(train_ct)}, Test CT : {len(test_ct)}")
    print(f"Train HIPPO : {len(train_hippo)}, Test HIPPO : {len(test_hippo)}")
    #print(len(train_edge), len(test_edge))

    train_frame = pd.DataFrame({'CT': train_ct,
                                'HIPPO' : train_hippo})
    
    test_frame = pd.DataFrame({'CT' : test_ct,
                               'HIPPO' : test_hippo})
    
    train_frame.to_excel(f"F:\\HIPPO\\train.xlsx", index=False)
    test_frame.to_excel(f"F:\\HIPPO\\test.xlsx", index=False)