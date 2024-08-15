from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from ngboost import NGBRegressor
import xgboost
from ngboost.distns import Normal
import sys



np.random.seed(123)


#High SNR case
# D = loadmat("YancoData_2021_All_validOnly_May13_5km_radius_SMAP_SNR10.mat",mat_dtype=True) 


#Low SNR case
D = loadmat("YancoData_2021_All_validOnly_May13_5km_radius_SMAP_SNR0.mat",mat_dtype=True) 




######################################

MM_all=D['MM_train']
SM_all=D['SM_train']
AUX_all=D['AUX_train']
SID_all=D['SID_train']
YR_all=D['YR_train']
SMAP_all=D['SMAP_train']


#AUX: sp_lat, sp_lon, sp_inc_angle,ddm_snr,clay_frac,sand_frac,silt_frac,elevation_m,dem_slope_deg,dem_aspect_deg



MM_all=MM_all/MM_all.max()



[n5,n6]=AUX_all.shape
mx2=np.zeros([n6])
for qq in range(n6):
    mx2[qq]=2*np.max(AUX_all[:,qq])
    AUX_all[:,qq]=np.divide(AUX_all[:,qq],mx2[qq])


[n1,n2,n3]=MM_all.shape
[n5,n6]=AUX_all.shape



#number of different training/validatiaon splits
IDX=1
    


acc_sel=np.zeros(IDX)
acc_sel_XG=np.zeros(IDX)
r_train = np.zeros(IDX)
r_XG_train = np.zeros(IDX)
r_XG_val= np.zeros(IDX)


r_val = np.zeros(IDX)
rmse_train = np.zeros(IDX)
bias_train = np.zeros(IDX)
unrmse_train = np.zeros(IDX)

rmse_val = np.zeros(IDX)
bias_val = np.zeros(IDX)
unrmse_val = np.zeros(IDX)

r_smap_train = np.zeros(IDX)
r_smap_val = np.zeros(IDX)
rmse_smap_train = np.zeros(IDX)
bias_smap_train = np.zeros(IDX)
unrmse_smap_train = np.zeros(IDX)

rmse_smap_val = np.zeros(IDX)
bias_smap_val = np.zeros(IDX)
unrmse_smap_val = np.zeros(IDX)


rmse_XG_val = np.zeros(IDX)
bias_XG_val = np.zeros(IDX)
unrmse_XG_val = np.zeros(IDX)

rmse_XG_train = np.zeros(IDX)
bias_XG_train = np.zeros(IDX)
unrmse_XG_train = np.zeros(IDX)


out_train_CROSS=[]
pred_train_CROSS=[]

pred_train_CROSS_std=[]

out_val_CROSS=[]
pred_val_CROSS=[]

pred_val_CROSS_std=[]

smap_train_CROSS=[]

smap_val_CROSS=[]



for idx in range(IDX):
    print('\n\n\n Realization '+str(idx))
    tmp=np.random.permutation(len(MM_all))
    num_train=int(np.round(0.8*len(MM_all)))
    
    train_set=tmp[0:num_train]
    
    num_val=len(MM_all)-num_train
    val_set=tmp[num_train:num_train+num_val]
    
    
    in_train_DDM=MM_all[train_set,:,:]
    in_train_DDM=np.expand_dims(in_train_DDM,axis=-1)
    aux_train=AUX_all[train_set,:]
    out_train=SM_all[train_set,:]
    smap_train=SMAP_all[train_set,:]
    
    VM_train=out_train>0
    
    
    
    in_val_DDM=MM_all[val_set,:,:]
    in_val_DDM=np.expand_dims(in_val_DDM,axis=-1)
    aux_val=AUX_all[val_set,:]
    out_val=SM_all[val_set,:]
    smap_val=SMAP_all[val_set,:]
    VM_val=out_val>0
    
    [n1,n2,n3,n4]=in_train_DDM.shape
    [n5,n6]=aux_train.shape
    
    [v1,v2,v3,v4]=in_val_DDM.shape
    
    num_train=len(in_train_DDM)
    num_test=len(in_val_DDM)
    
    
    in_train_DDM=np.reshape(in_train_DDM,[num_train,n2*n3])
    
    in_val_DDM=np.reshape(in_val_DDM,[num_test,n2*n3])
    
    in_train=np.concatenate((in_train_DDM,aux_train),axis=1)
    in_val=np.concatenate((in_val_DDM,aux_val),axis=1)   
    
    
    
    model_NGBoost=NGBRegressor(Dist=Normal, n_estimators=1000)
    model_XGBoost=xgboost.XGBRegressor(n_estimators=1000,booster='gbtree')
    
    
    pred_train_lst=np.zeros([n1,1])
    pred_train_lst_XG=np.zeros([n1,1])
    pred_val_lst=np.zeros([v1,1])
    pred_val_NG_std=np.zeros([v1,1])
    pred_val_lst_XG=np.zeros([v1,1])
    
    pred_train_lst_std=np.zeros([n1,1])
    pred_val_lst_std=np.zeros([v1,1])
    
  
    
 
    model_NGBoost.fit(in_train, out_train)
    model_XGBoost.fit(in_train, out_train)
 
    pred_train_XG=model_XGBoost.predict(in_train)
    Y_dists_train=model_NGBoost.pred_dist(in_train)  
    pred_train_NG=Y_dists_train.params['loc']
    pred_train_NG_std=Y_dists_train.params['scale']

    
    pred_val_XG=model_XGBoost.predict(in_val)  
    Y_dists_val=model_NGBoost.pred_dist(in_val)  
    pred_val_NG=Y_dists_val.params['loc']
    pred_val_NG_std=Y_dists_val.params['scale']
    
    # Compute the prediction intervals for each test example
    alpha = 0.05  # 95% confidence interval
    y_lower_train, y_upper_train = Y_dists_train.interval(alpha)
    y_lower_val, y_upper_val = Y_dists_val.interval(alpha)
    
     
    alph=1.0
    coverage_train = np.mean((out_train >= (pred_train_NG-alph*pred_train_NG_std)) & (out_train <= (pred_train_NG+alph*pred_train_NG_std)))
    coverage_val  = np.mean((out_val >= (pred_val_NG-alph*pred_val_NG_std)) & (out_val <= (pred_val_NG+alph*pred_val_NG_std)))
    print("Coverage probability (tain):", coverage_train)
    print("Coverage probability (val):", coverage_val)
    

    alph=1.65
    coverage_train = np.mean((out_train >= (pred_train_NG-alph*pred_train_NG_std)) & (out_train <= (pred_train_NG+alph*pred_train_NG_std)))
    coverage_val  = np.mean((out_val >= (pred_val_NG-alph*pred_val_NG_std)) & (out_val <= (pred_val_NG+alph*pred_val_NG_std)))
    print("Coverage probability (tain):", coverage_train)
    print("Coverage probability (val):", coverage_val)


    alph=1.96
    coverage_train = np.mean((out_train >= (pred_train_NG-alph*pred_train_NG_std)) & (out_train <= (pred_train_NG+alph*pred_train_NG_std)))
    coverage_val  = np.mean((out_val >= (pred_val_NG-alph*pred_val_NG_std)) & (out_val <= (pred_val_NG+alph*pred_val_NG_std)))
    print("Coverage probability (tain):", coverage_train)
    print("Coverage probability (val):", coverage_val)
    
    alph=2.58
    coverage_train = np.mean((out_train >= (pred_train_NG-alph*pred_train_NG_std)) & (out_train <= (pred_train_NG+alph*pred_train_NG_std)))
    coverage_val  = np.mean((out_val >= (pred_val_NG-alph*pred_val_NG_std)) & (out_val <= (pred_val_NG+alph*pred_val_NG_std)))
    print("Coverage probability (tain):", coverage_train)
    print("Coverage probability (val):", coverage_val)

   
    
    
    z_score_train=(out_train[:,0]-pred_train_NG)/pred_train_NG_std
    
   
    z_score_train_counts, z_score_train_bins = np.histogram(z_score_train,100,range=[-10,10],density=True)
    plt.hist(z_score_train_bins[:-1], z_score_train_bins, weights=z_score_train_counts,label='Model',alpha=1.0)
    plt.title('Histogram of z-scores (train)')
    # plt.show()
    
    from scipy.stats import norm
    x = np.arange(-10, 10, 0.001) # range of x in spec
    y = norm.pdf(x,0,1)
    plt.plot(x,y)
    
    plt.show()

    z_score_val=(out_val[:,0]-pred_val_NG)/pred_val_NG_std
    
    # z_score_train_hist=np.histogram(z_score_train,100,range=[-10,10])
    
    z_score_val_counts, z_score_val_bins = np.histogram(z_score_val,100,range=[-10,10],density=True)
    plt.hist(z_score_val_bins[:-1], z_score_val_bins, weights=z_score_val_counts,alpha=1.0,label='Experimental')
    #plt.title('Histogram of z-scores (val)')
    # plt.show()
    
    
    plt.plot(x,y,label='Theoretical')
    plt.legend()
    

    
    plt.show()    
    
    
  
        
  
    pred_train_std_1=1*np.std(pred_train_lst,axis=-1)
    pred_val_std_1=1*np.std(pred_val_lst,axis=-1)
    
    
    auc=pred_val_std_1-np.abs(pred_val_NG-out_val[:,0])
    auc_sel=np.where(auc>=0)[0]
    # acc_sel[idx]=len(auc_sel)/len(pred_val_NG)
    
    auc_XG=pred_val_std_1-np.abs(pred_val_NG-out_val[:,0])
    auc_sel_XG=np.where(auc_XG>=0)[0]
    # acc_sel_XG[idx]=len(auc_sel_XG)/len(pred_val_XG)
    
    # print(acc_sel[idx])
    
    true_train = out_train
    true_val = out_val
    
    x=np.linspace(0, n1, n1)
    y=pred_val_NG
    yerr=pred_val_std_1


    err_NG_boost=np.abs(pred_val_NG.flatten()-out_val.flatten())    
   
        


    a=(out_val.flatten()>(pred_val_NG.flatten()-pred_val_NG_std.flatten())).astype(int)
    b=(out_val.flatten()<(pred_val_NG.flatten()+pred_val_NG_std.flatten())).astype(int)
    c=a&b
    d=np.sum(c)/len(out_val)
    print(d)
    
    a=np.sum(-np.log(pred_val_NG_std.flatten()))
    b=np.sum(np.abs(pred_val_NG.flatten()-out_val.flatten()))
    print(a,b)
 
    
    a=np.argsort(true_val.flatten())
    b=np.sort(true_val.flatten())
    
    mean_pred_val_NG_std=np.mean(pred_val_NG_std)
    plt.axhline(y = mean_pred_val_NG_std, color = 'r', linestyle = '-')
    
    plt.plot(b,pred_val_NG_std[a],'bx')
    plt.xlabel('In-situ SM  ' + r'$(cm^3/cm^3)$')
    plt.ylabel('prediction uncertainty (std)')
    plt.ylim([0,0.1])
        
    plt.show()

 
    
    counts_IS, bins_IS = np.histogram(out_val,100,(0.001,0.4))
    plt.hist(bins_IS[:-1], bins_IS, weights=counts_IS,label='In Situ',alpha=0.7,rwidth=0.5)
    
    counts_PR, bins_PR = np.histogram(pred_val_NG,100,(0.001,0.4))
    plt.hist(bins_PR[:-1], bins_PR, weights=counts_PR,label='NGboost',alpha=0.7)
    
    counts_XG, bins_XG = np.histogram(pred_val_XG,100,(0.001,0.4))
    plt.hist(bins_XG[:-1], bins_XG, weights=counts_XG,label='XGboost',alpha=0.7)
    
    plt.xlim([0,0.4])
    # plt.title('Histogram of SM values')
    plt.xlabel('SM ' + r'$(m^3/m^3)$')
    plt.ylabel('Number of samples')
    plt.legend()
    # name='res_12_2022\Histogram of SM values'+ str(scenario) + '_k_' +str(K) +   '.pdf'
    # plt.savefig(name,dpi=600)
    plt.show()

 
    
    r_train[idx]=np.mean((true_train.flatten()-np.mean(true_train.flatten()))*(pred_train_NG.flatten()-np.mean(pred_train_NG.flatten())))/(np.std(true_train.flatten())*np.std(pred_train_NG.flatten()))
    r_val[idx]=np.mean((true_val.flatten()-np.mean(true_val.flatten()))*(pred_val_NG.flatten()-np.mean(pred_val_NG.flatten())))/(np.std(true_val.flatten())*np.std(pred_val_NG.flatten()))
    
    r_XG_train[idx]=np.mean((true_train.flatten()-np.mean(true_train.flatten()))*(pred_train_XG.flatten()-np.mean(pred_train_XG.flatten())))/(np.std(true_train.flatten())*np.std(pred_train_XG.flatten()))
    r_XG_val[idx]=np.mean((true_val.flatten()-np.mean(true_val.flatten()))*(pred_val_XG.flatten()-np.mean(pred_val_XG.flatten())))/(np.std(true_val.flatten())*np.std(pred_val_XG.flatten()))
    
    
    r_smap_train[idx]=np.mean((true_train.flatten()-np.mean(true_train.flatten()))*(smap_train.flatten()-np.mean(smap_train.flatten())))/(np.std(true_train.flatten())*np.std(smap_train.flatten()))
    r_smap_val[idx]=np.mean((true_val.flatten()-np.mean(true_val.flatten()))*(smap_val.flatten()-np.mean(smap_val.flatten())))/(np.std(true_val.flatten())*np.std(smap_val.flatten()))
    
    
    # r_train=np.corrcoef(true_train.flatten(), pred_train.flatten())
    # r2_train=r2_score(true_train.flatten(), pred_train.flatten())
    print('R train', r_train[idx])
    
    # r_val=np.corrcoef(true_val.flatten(), pred_val.flatten())
    print('R val', r_val[idx])
    
    rmse_train[idx]=np.math.sqrt(sklearn.metrics.mean_squared_error(true_train.flatten(), pred_train_NG.flatten()))
    bias_train[idx]=np.mean(true_train.flatten()) - np.mean(pred_train_NG.flatten())
    unrmse_train[idx]=rmse_train[idx]-bias_train[idx]
    # print('RMSE train', rmse_train[idx])
    # print('Bias train',bias_train[idx])
    # print('unRMSE train', unrmse_train[idx])
    
    
    rmse_smap_train[idx]=np.math.sqrt(sklearn.metrics.mean_squared_error(true_train.flatten(), smap_train.flatten()))
    bias_smap_train[idx]=np.mean(true_train.flatten()) - np.mean(smap_train.flatten())
    unrmse_smap_train[idx]=rmse_smap_train[idx]-bias_smap_train[idx]
    # print('RMSE SMAP train', rmse_smap_train[idx])
    # print('Bias SMAP train',bias_smap_train[idx])
    # print('unRMSE SMAP train', unrmse_smap_train[idx])
    
    
    rmse_XG_train[idx]=np.math.sqrt(sklearn.metrics.mean_squared_error(true_train.flatten(), pred_train_XG.flatten()))
    bias_XG_train[idx]=np.mean(true_train.flatten()) -  np.mean(pred_train_XG.flatten())
    unrmse_XG_train[idx]=rmse_XG_train[idx]-bias_XG_train[idx]
    # print('RMSE XG train', rmse_XG_train[idx])
    # print('Bias XG train',bias_XG_train[idx])
    # print('unRMSE XG train', unrmse_XG_train[idx])
    
    rmse_val[idx]=np.math.sqrt(sklearn.metrics.mean_squared_error(true_val.flatten(), pred_val_NG.flatten()))
    bias_val[idx]=np.mean(true_val.flatten()) -  np.mean(pred_val_NG.flatten())
    unrmse_val[idx]=rmse_val[idx]-bias_val[idx]
    # print('RMSE val', rmse_val[idx])
    # print('Bias val',bias_val[idx])
    # print('unRMSE val', unrmse_val[idx])
    
    
    rmse_smap_val[idx]=np.math.sqrt(sklearn.metrics.mean_squared_error(true_val.flatten(), smap_val.flatten()))
    bias_smap_val[idx]=np.mean(true_val.flatten()) -  np.mean(smap_val.flatten())
    unrmse_smap_val[idx]=rmse_smap_val[idx]-bias_smap_val[idx]
    # print('RMSE SMAP val', rmse_smap_val[idx])
    # print('Bias SMAP val',bias_smap_val[idx])
    # print('unRMSE SMAP val', unrmse_smap_val[idx])
    
    
    rmse_XG_val[idx]=np.math.sqrt(sklearn.metrics.mean_squared_error(true_val.flatten(), pred_val_XG.flatten()))
    bias_XG_val[idx]=np.mean(true_val.flatten()) -  np.mean(pred_val_XG.flatten())
    unrmse_XG_val[idx]=rmse_XG_val[idx]-bias_XG_val[idx]
    # print('RMSE XG val', rmse_XG_val[idx])
    # print('Bias XG val',bias_XG_val[idx])
    # print('unRMSE XG val', unrmse_XG_val[idx])

    
    
    
    out_train_CROSS.append(out_train)
    pred_train_CROSS.append(pred_train_NG)
    smap_train_CROSS.append(smap_train)
    
    pred_train_CROSS_std.append(pred_train_NG_std)
    
    out_val_CROSS.append(out_val)
    pred_val_CROSS.append(pred_val_NG)
    smap_val_CROSS.append(smap_val)
    
    pred_val_CROSS_std.append(pred_val_NG_std)
    
    
    
  






    
print('\n\nAverage')

r_train_mn=np.mean(r_train)
r_train_std=np.std(r_train)
r_val_mn=np.mean(r_val)
r_val_std=np.std(r_val)
rmse_train_mn=np.mean(rmse_train)
rmse_train_std=np.std(rmse_train)
rmse_val_mn=np.mean(rmse_val)
rmse_val_std=np.std(rmse_val)
unrmse_train_mn=np.mean(unrmse_train)
unrmse_train_std=np.std(unrmse_train)
unrmse_val_mn=np.mean(unrmse_val)
unrmse_val_std=np.std(unrmse_val)


smap_r_train_mn=np.mean(r_smap_train)
smap_r_train_std=np.std(r_smap_train)
smap_r_val_mn=np.mean(r_smap_val)
smap_r_val_std=np.std(r_smap_val)
smap_rmse_train_mn=np.mean(rmse_smap_train)
smap_rmse_train_std=np.std(rmse_smap_train)
smap_rmse_val_mn=np.mean(rmse_smap_val)
smap_rmse_val_std=np.std(rmse_smap_val)
smap_unrmse_train_mn=np.mean(unrmse_smap_train)
smap_unrmse_train_std=np.std(unrmse_smap_train)
smap_unrmse_val_mn=np.mean(unrmse_smap_val)
smap_unrmse_val_std=np.std(unrmse_smap_val)


XG_r_train_mn=np.mean(r_XG_train)
XG_r_train_std=np.std(r_XG_train)
XG_r_val_mn=np.mean(r_XG_val)
XG_r_val_std=np.std(r_XG_val)
XG_rmse_train_mn=np.mean(rmse_XG_train)
XG_rmse_train_std=np.std(rmse_XG_train)
XG_rmse_val_mn=np.mean(rmse_XG_val)
XG_rmse_val_std=np.std(rmse_XG_val)
XG_unrmse_train_mn=np.mean(unrmse_XG_train)
XG_unrmse_train_std=np.std(unrmse_XG_train)
XG_unrmse_val_mn=np.mean(unrmse_XG_val)
XG_unrmse_val_std=np.std(unrmse_XG_val)



print('\n')
print('RMSE NG train', rmse_train_mn,rmse_train_std)
print('unRMSE NG train', unrmse_train_mn,unrmse_train_std)
print('R NG train', r_train_mn,r_train_std)
print('\n')
print('RMSE XG train', XG_rmse_train_mn,XG_rmse_train_std)
print('unRMSE XG train', XG_unrmse_train_mn,XG_unrmse_train_std)
print('R XG train', XG_r_train_mn,XG_r_train_std)
print('\n')
print('RMSE SMAP train', smap_rmse_train_mn,smap_rmse_train_std)
print('unRMSE SMAP train', smap_unrmse_train_mn,smap_unrmse_train_std)
print('R SMAP train', smap_r_train_mn,smap_r_train_std)
print('\n')
print('RMSE NG val', rmse_val_mn,rmse_val_std)
print('unRMSE NG val', unrmse_val_mn,unrmse_val_std)
print('R NG val', r_val_mn,r_val_std)
print('\n')
print('RMSE XG val', XG_rmse_val_mn,XG_rmse_val_std)
print('unRMSE XG val', XG_unrmse_val_mn,XG_unrmse_val_std)
print('R XG val', XG_r_val_mn,XG_r_val_std)
print('\n')
print('RMSE SMAP val', smap_rmse_val_mn,smap_rmse_val_std)
print('unRMSE SMAP val', smap_unrmse_val_mn,smap_unrmse_val_std)
print('R SMAP val', smap_r_val_mn,smap_r_val_std)


# for ii in range(K):
plt.scatter(out_train_CROSS[0],pred_train_CROSS[0],s=3.0,label='Proposed',color = 'red', alpha=0.75)
plt.scatter(out_train_CROSS[0],smap_train_CROSS[0],s=3.0,label='Baseline',color = 'green', alpha=0.75)
plt.scatter(np.arange(0,0.5,0.01),np.arange(0,0.5,0.01),s=1.0,color = 'blue')
plt.legend(loc='upper left')
plt.xlabel('Measured (In-Situ)')
plt.ylabel('Predicted')
# txt="R = {:.2f}"
# plt.text(0.35,0.08,txt.format(r_train_mn))
# txt="RMSE = {:.4f}"
# plt.text(0.35,0.05,txt.format(rmse_train_mn))
# txt="ubRMSE = {:.4f}"
# plt.text(0.35,0.02,txt.format(unrmse_train_mn))
# plt.legend(loc='upper left')
plt.title('Training')

  

plt.show()


# for ii in range(K):
plt.scatter(out_val_CROSS[0],pred_val_CROSS[0],s=3.0,label='Proposed',color = 'red', alpha=0.75)
plt.scatter(out_val_CROSS[0],smap_val_CROSS[0],s=3.0,label='Baseline',color = 'green', alpha=0.75)    
plt.scatter(np.arange(0,0.5,0.01),np.arange(0,0.5,0.01),s=1.0,color = 'blue')
plt.legend(loc='upper left')
plt.xlabel('Measured (In-Situ)')
plt.ylabel('Predicted')
# txt="R = {:.2f}"
# plt.text(0.35,0.08,txt.format(r_val_mn))
# txt="RMSE = {:.4f}"
# plt.text(0.35,0.05,txt.format(rmse_val_mn))
# txt="ubRMSE = {:.4f}"
# plt.text(0.35,0.02,txt.format(unrmse_val_mn))
# plt.legend(loc='upper left')
plt.title('Validation')


plt.show()



