import torch
from collections import OrderedDict

weights_files1 = "/media/disk3/lmy/mask2former/checkpoint/sam_vit_h_4b8939_rename.pth"
weights_files2 = "/media/disk3/lmy/centernet-better-adapter/outputs/playground/vit_sam_adapter_1024_whu_mix_0208/1/model_0013999.pth"

weights1 = torch.load(weights_files1)
# for k, v in weights1.items():  # key, value
#     print(k)  # 打印 key（参数名）

print("______________________________")

weights2 = torch.load(weights_files2)
weights2 = weights2["model"]
# for k, v in weights2.items():  # key, value
#     print(k)  # 打印 key（参数名）

new_state_dict = OrderedDict()
for k in weights1:
    new_state_dict[k] = weights1[k]
    print(k)

print("__________________________")

for j in weights2:
    if j not in new_state_dict.keys():
        new_state_dict[j] = weights2[j]
        print(j)

print("==============================")
for k, v in new_state_dict.items():  # key, value
    print(k)
torch.save(new_state_dict,
           '/media/disk3/lmy/centernet-better-adapter-sam/checkpoint/sam_centernet_pretrained_whu_mix_0208.pth')