import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from scipy.interpolate import make_interp_spline
import numpy as np

p1=plt.figure(figsize=(14,6),dpi=300) #第一幅子图,并确定画布大小
# p1.suptitle('loss',fontsize = 14, fontweight='bold')
# plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)  #位置调整

ax1=p1.add_subplot(1,2,1)
net = pd.read_csv(r'C:\Users\xihewen\Desktop\lunwen\CSV\liver-att\34\run-34-tag-loss.csv', usecols=['Step', 'Value'])
Step = np.linspace(net.Step.min(),net.Step.max(),900)
Value = make_interp_spline(net.Step,net.Value)(Step)
net3 = pd.read_csv(r'C:\Users\xihewen\Desktop\lunwen\CSV\liver-att\34\tou.csv', usecols=['Step', 'Value'])
Step1 = np.linspace(net3.Step.min(),net3.Step.max(),900)
Value1 = make_interp_spline(net3.Step,net3.Value)(Step1)
plt.plot(Step , Value, lw=1.5, label='Att-DialResUNet3D without attention', color='green')
plt.plot(Step1, Value1, lw=1.5, label='Att-DialResUNet3D with attention', color='pink')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Liver')
plt.legend(loc=0)

ax2=p1.add_subplot(1,2,2)
net2 = pd.read_csv(r'C:\Users\xihewen\Desktop\lunwen\CSV\tumor_att\run-13-tag-loss.csv', usecols=['Step', 'Value'])
net3 = pd.read_csv(r'C:\Users\xihewen\Desktop\lunwen\CSV\tumor_noatt\run-12-tag-loss.csv', usecols=['Step', 'Value'])
Step = np.linspace(net2.Step.min(),net2.Step.max(),800)
Value = make_interp_spline(net2.Step,net2.Value)(Step)
Step1 = np.linspace(net3.Step.min(),net3.Step.max(),800)
Value1 = make_interp_spline(net3.Step,net3.Value)(Step1)
plt.plot(Step1 , Value1, lw=1.5, label='Att-DialResUNet3D without attention', color='green')
plt.plot(Step, Value, lw=1.5, label='Att-DialResUNet3D with attention', color='pink')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Tumor')
plt.legend(loc=0)

plt.savefig(r'C:\Users\xihewen\Desktop\lunwen\images\loss.tif',dpi=300)
plt.show()
