首先，我的初始情况是两个SSD和一个HDD，Win10系统盘装在了其中一个SSD上，D盘用掉了HDD一半的空间，内存16G，GPU RTX 2080(万恶之源).因为RTX对应的显卡驱动是410，百度的所有用apt-get的方法都不能直接安装，需要手动下载。并且410对应的是cuda10.0,非常多简便的操作都无法直接实现。




1.官网下载ubuntu16.04（desktop版本，后缀为.iso的文件），使用ultraISO（下载试用版就行）将U盘制作为启动盘

1.1 文件---打开.iso文件

1.2 启动---写入硬盘映像

1.3 若安装在HDD上，制作方式选USB-HDD+，安在SSD上，选RAW。





2.在开机时进入BIOS界面，关闭快速启动，在启动顺序那里把U盘设为第一个，然后F10（保存并退出）。等它重启就好。





3.选择Install Ubuntu， 语言选英文，不然命令行操作会有问题，我没有联网，选了安装第三方，在installation type选something else。





4.分区选项：我用的是256的SSD和1T的HDD

SSD：

           /boot  主分区      250MB      Ext4

           swap  逻辑分区  16GB（等于内存）

            /     主分区     剩下所有      Ext4
HDD：

           /home  主分区     1T             Ext4





5.系统装好后，这时候应该是分辨率很低看起来很恶心的状态。直接进入nvidia官网，选出自己的显卡型号，下载相应的.run格式的驱动文件，放在Downloads下。

5.1 先（sudo apt-get remove --purge nvidia*）卸载原来的驱动

5.2 禁用nouveau：先（sudo gedit /etc/modprobe.d/blacklist.conf）打开一个文件，在最后一行添加（blacklist nouveau）,之后在命令行（sudo update-initramfs -u）

5.3 重启电脑后 （lsmod | grep nouveau），如果没有输出说明禁用成功





6.安装新的显卡驱动

6.1  Hit （Ctrl+Alt+F1） and login using your credentials.

6.2 kill your current X server session by typing （sudo service lightdm stop） or （sudo lightdm stop）

6.3 Enter runlevel 3 by typing （sudo init 3）

6.4 Install your *.run file.you change to the directory where you have downloaded the file by typing for instance cd Downloads. If it is in another directory, go there. Check if you see the file when you type （ls NVIDIA*）
Make the file executable with （chmod +x ./your-nvidia-file.run）
Execute the file with （sudo ./your-nvidia-file.run -no-opengl-files）

6.5 You might be required to reboot when the installation finishes. If not, run sudo service lightdm start or sudo start lightdm to start your X server again.

6.6 It's worth mentioning, that when installed this way, you'd have to redo the steps after each kernel update

6.7 如果显卡驱动安装成功了，重启之后分辨率就会自动调整到很高的状态了。




7.安装Anaconda+Pycharm

7.1 下载Pycharm（免费的Community版本）到Downloads文件夹

7.2 (cd Downloads) 解压缩(tar xfz pycharm-*.tar.gz) 移动到目标文件夹（sudo mv pycharm-* /usr/local）进入目标bin文件夹（cd /usr/local/pycharm-*/bin/）运行文件（./pycharm.sh）接下来就是图形化界面了

7.3 将Pycharm锁定在启动行，不然没有快捷方式

7.4 官网下载Anaconda，我下载的版本是Anaconda3-5.2.0-Linux-x86_64.sh（这个版本是默认Python3.6的）

7.5 安装（sudo bash Downloads/Anaconda3-5.2.0-Linux-x86_64.sh）安装完成后重载一下文件（source ~/.bashrc）

7.6 接下来打开Pycharm，File—settings—Project—Project Interpreter—右边的小齿轮—add—Existing Environment—找到anaconda
3的bin下面的python3.6就好，这样就可以在Pycharm里面用anaconda的东西了。




8.安装Cuda10.0（如果用2080的卡，默认的410驱动必须要cuda10.0）

8.1 先安装一下相关的包，我也不知道是干啥的（sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev  ）

8.2 RTX2080要用410的驱动，这个驱动只支持Cuda10.0，所以到官网下载Cuda10.0， 选择Linux—x86_64—Ubuntu—16.04—runfile(local), 把Base下了

8.3 （sudo sh cuda_10.0.130_410.48_linux.run）安装cuda，问要不要安装Nvidia Graphic Driver的时候一定要选no，不然就回到解放前了。

8.4  如果一切顺利，这时候/usr/local 文件夹里会出现一个叫cuda10.0的文件夹，接下来就是添加路径

8.5

       （sudo gedit /etc/profile）打开一个文本，在文本的最后加上两行：
       （export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}）
       （export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}）

8.6  添加完成后重启电脑，在命令行输入nvcc -V，如果显示出来cuda的版本啥的就说明cuda安装成功了





9.安装cuDNN

9.1  官网注册并且下载，要下载和自己安装的cuda版本相配的cuDNN版本，不然会有问题。我下载的是（cudnn-10.0-linux-x64-v7.3.1.20.tgz）

9.2 (cd Downloads) 解压缩( tar -xzvd cudnn-10.0-linux-x64-v7.3.1.20.tgz)

9.3 复制到到相应位置

           （sudo cp cuda/include/cudnn.h /usr/local/cuda-10.0/include）
            (sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64）
            (sudo chmod a+r /usr/local/cuda-10.0/include/cudnn* /usr/local/cuda-10.0/lib64/libcudnn*)






10.安装Pytorch 和 torchvision（cuda10.0对应的版本似乎只能从github上装）

10.1 装一下git（sudo apt-get install git）

10.2 安装一些包:先sudo chmod -R 777 (anaconda的安装目录):很重要，不然会没有权限

      （export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" ）
      （conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing）
      （conda install -c mingfeima mkldnn）

10.3 git clone --recursive https://github.com/pytorch/pytorch

10.4 （cd pytorch） （python setup.py install）

10.5 同样的方式 git clone --recursive https://github.com/pytorch/vision

10.6 （cd vision） （python setup.py install）

如果足够幸运，这时候应该可以用pytorch在gpu上计算了









