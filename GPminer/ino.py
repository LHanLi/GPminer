import dill,os,pickle,datetime


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

# log 函数
def log(*txt):
    try:
        #try:
        #    f = open('log.txt','a+', encoding='gbk')
        #except:
        #    f = open('log.txt','a+', encoding='utf-8')
        f = open('log.txt','a+', encoding='gbk')
        write_str = ('\n'+' '*35).join([str(i) for i in txt])
        f.write('%s,        %s\n' % \
            (datetime.datetime.now(), write_str))
        f.close()
    except PermissionError as e:
        print(f"Error: {e}. You don't have permission to access the specified file.")
