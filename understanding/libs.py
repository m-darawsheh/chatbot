


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


# x = torch.tensor([1.0, 2.0])
# y = torch.tensor([3.0, 4.0])
# z = x + y
# print (z)

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


# methods of tensor
# Tensor صفرية
# t2 = torch.zeros(3)
# print(t2)

# Tensor كلها ones
# t3 = torch.ones(3)
# g = torch.ones([3, 4, 5])
# evrey once you append params to the tensor will be add dimnsion in g (3, 4, 5) 3D 
# print(t3)
# print(g)

# new = torch.zeros([3, 4, 5])
# print(new)

# Tensor عشوائي
# t4 = torch.rand(3)
# print(t4)


# operations on tensor
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])
# print(a + b)        # add
# print(a - b)        # minus
# print(a * b)        # multiply
# print(a / b)        # divide
# print(torch.dot(a, b))  # dot product i think you remember dot product from the math class
# print(torch.cross(a, b))  # cross product i think you remember cross product from the math class


# reshape tensor
# x = torch.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
# print(x)
# reshaped = x.view(2, 3)  # 2x3
# print(reshaped)


# concatinate tensor
# x = torch.tensor([[1, 2], [3, 4]])  # 2x2
# y = torch.tensor([[5, 6], [7, 8]])  # 2x2
# z = torch.cat((x, y), dim=0)  # 4x2
# print(z)
# dim=0 it means concatinate the rows
# dim=1 it means concatinate the columns
# u = torch.tensor([[1, 2], [3, 4]])  # 2x2
# v = torch.tensor([[5, 6], [7, 8]])  # 2x2
# y = torch.cat((u, v), dim=1)  # 4x2
# print(y)


# compare tensor
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([2, 2, 2])

# print(a == b)   # compare
# print(a > b)    # greater than
# you musttttttttttt see the result of this operation !!!!!!!!!!


# comvert list to tensor
# a = [1, 2, 3]
# b = torch.tensor(a)

# from tensor to list
# a = torch.tensor([1, 2, 3])
# b = a.tolist()