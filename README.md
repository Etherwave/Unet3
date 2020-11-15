# Unet3

Unet3复现

由于个人设备原因，参数相对于原论文的网络削减了。

每层的channel由

filters = [64, 128, 256, 512, 1024]

改为了

filters = [32, 64, 128, 256, 512]

分割数据是使用的camvid

下载链接：

https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz



