


# to install the libraries do the following commands in the terminal
# {
#   pip install torch
#   pip install transformers
#   pip install flask
# }

# اسمها بالعربي الشعله
import torch   

# this lib very strong in AI and ML(Machine Learning) and DL(Deep Learning)
# this lib work with GPU and CPU (does not matter) => search about this and tell me what the idea from it
# this lib work with tensor (i will explain it later) , it is like data type in python , and can do math operations on tensor


x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0])
print(x + y)
# the result is tensor([4.0, 6.0])

# now you must tell me this not a tensor, it is a list
# but in the background it is a tensor
# # and you can do math operations on it
# | الاسم   الآخر   | الشكل               | مثال                 | الاسم    |
# | ---------- | ------------------- | -------------------- | ------------- |
# | **Scalar** | رقم واحد            | `5`                  | قيمة مفردة    |
# | **Vector** | قائمة من الأرقام    | `[1, 2, 3]`          | متجه          |
# | **Matrix** | جدول (صفوف × أعمدة) | `[[1, 2], [3, 4]]`   | مصفوفة        |
# | **Tensor** | أي عدد من الأبعاد   | `[[[1, 2], [3, 4]]]` | متعدد الأبعاد |


# Scalar (عدد مفرد)
s = torch.tensor(5)

# Vector (متجه)
v = torch.tensor([1, 2, 3])

# Matrix (مصفوفة 2x2)
m = torch.tensor([[1, 2], [3, 4]])

# Tensor (3D)
t = torch.tensor([[[1, 2], [3, 4]]])
# Tensor (4D), (5D) .... got it ???????????

# why we use tensor because it can hold any data type (image, text, audio, video)
# for image i think we will use tensor in the second project
# for text we will use tensor in this project but we have to convert the text to tensor 
# who can do this ??????????
# the answer is the transformer library or convert the text to tensor manually by transform the text to numbers and then convert it to tensor