import sys
import os
from shutil import copyfile
import platform
from glob import glob

if len(sys.argv) != 14:
  print('please input the right parameters')
  sys.exit(1)
current_os_name = platform.platform()
print('%s' % current_os_name)

if 'Ubuntu' in current_os_name.split('-'): #on local
  sysflag='local'
elif 'centos' in current_os_name.split('-'): #on lewis or multicom
  sysflag='lewis'

GLOBAL_PATH=os.path.dirname(os.path.dirname(__file__)) #this will auto get the DNCON4 folder name

sys.path.insert(0, GLOBAL_PATH+'/lib/')
print (GLOBAL_PATH)
from Model_training import *
from training_strategy import *
import global_para 
from generate_feature import cal_feature_num

net_name = str(sys.argv[1]) # 
dataset = str(sys.argv[2])  # 
fea_file = str(sys.argv[3])
predict_method = str(sys.argv[4]) # realdist_hdist realdist_hdist_nointra
nb_filters=int(sys.argv[5]) 
nb_layers=int(sys.argv[6]) 
filtsize=int(sys.argv[7]) 
out_epoch=int(sys.argv[8])
in_epoch=int(sys.argv[9]) 
feature_dir = sys.argv[10] 
outputdir = sys.argv[11] 
acclog_dir = sys.argv[12]
index = float(sys.argv[13])


CV_dir=outputdir+'/'+net_name+'_'+dataset+'_'+fea_file + '_' + predict_method + '_filter'+str(nb_filters)+'_layers'+str(nb_layers)+'_ftsize'+str(filtsize)+'_'+str(index)

lib_dir=GLOBAL_PATH+'/lib/'

# gpu_schedul_strategy(sysflag, gpu_mem_rate = 0.8, allow_growth = False)
# global_para._init()
# global_para.set_value('cuda', cuda_num)
  
rerun_epoch=0
if not os.path.exists(CV_dir):
  os.makedirs(CV_dir)
else:
  h5_num = len(glob(CV_dir + '/model_weights/*.h5'))
  rerun_epoch = h5_num
  if rerun_epoch <= 0:
    rerun_epoch = 0
    print("This parameters already exists, quit")
    # sys.exit(1)
  print("####### Restart at epoch ", rerun_epoch)

def chkdirs(fn):
  dn = os.path.dirname(fn)
  if not os.path.exists(dn): os.makedirs(dn)

def chkfiles(fn):
  if os.path.exists(fn):
    return True 
  else:
    return False

dist_string = '80'
if dataset == 'homodimer':
  path_of_lists   = f'{GLOBAL_PATH}/example/training_datalists/homodimer/'
elif dataset == 'heterodimer':
  path_of_lists   = f'{GLOBAL_PATH}/example/training_datalists/heterodimer/'
else:
  #add your dataset path
  print('Please input the dataset parameter!')
  sys.exit(1)
reject_fea_file = GLOBAL_PATH+'/lib/feature_txt/'+fea_file+'.txt'
path_of_Y       = feature_dir 
path_of_X       = feature_dir

if not os.path.exists(path_of_X):
  print("Can not find folder of features: "+ path_of_X +", please check and run configure.py to download or extract it!")
  sys.exit(1)

Maximum_length=450 

import time
print("Maximum_length % d"%Maximum_length)
start_time = time.time()

feature_num = cal_feature_num(reject_fea_file)

best_acc=HomoPred_train(feature_num, CV_dir, net_name, out_epoch, in_epoch, rerun_epoch, filtsize,
  nb_filters, nb_layers, 1, path_of_lists, path_of_Y, path_of_X, Maximum_length, reject_fea_file, predict_method=predict_method,
  if_use_binsize = False, rate = 1.0) #True

model_prefix = net_name
acc_history_out = "%s/%s.acc_history" % (acclog_dir, model_prefix)
chkdirs(acc_history_out)
if chkfiles(acc_history_out):
    print ('acc_file_exist,pass!')
    pass
else:
    print ('create_acc_file!')
    with open(acc_history_out, "w") as myfile:
        myfile.write("time\t netname\t filternum\t layernum\t kernelsize\t batchsize\t accuracy\n")

time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
acc_history_content = "%s\t %s\t %s\t %s\t %s\t %s\t %.4f\n" % (time_str, model_prefix, str(nb_filters),str(nb_layers),str(filtsize),str(1),best_acc)
with open(acc_history_out, "a") as myfile: myfile.write(acc_history_content) 
print("--- %s seconds ---" % (time.time() - start_time))
print("outputdir:", CV_dir)