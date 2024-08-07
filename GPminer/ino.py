import dill,os,pickle


# 反序列化一个对象

def save_pkl(obj,filename, fileloc, serial=False):
    if serial:
        obj = dill.dumps(obj)
    if type(obj)==type(set()):
        type_name = '.set'
    elif type(obj)==type(list()):
        type_name = '.list'
    elif type(obj)==type(dict()):
        type_name = '.dict'
    with open(fileloc+'/'+filename+type_name+'.pkl','wb') as file:
        pickle.dump(obj,file)
def read_pkl(filename, fileloc, serial=False):
    filename = [i for i in os.listdir(fileloc) if filename in i][0]
    with open(fileloc+'/'+filename,'rb') as file:
        obj = pickle.load(file)
        if serial:
            obj = dill.loads(obj)
        return obj
