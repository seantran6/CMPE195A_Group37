import torch

checkpoint_path = "gender_model_final.pth"
checkpoint = torch.load(checkpoint_path)

# If it's a full model
# print(checkpoint)

# If it's a state_dict only
print("All keys in the state_dict:")
for key in checkpoint.keys():
    print(key)
