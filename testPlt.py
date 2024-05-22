# def function(**kwargs):
#     print(kwargs['name'])

# # function(name='abcd', city='123')
    
# def add(*args, **abc):
#     print(args)
#     print(abc)

# add(1, 'ads', [1,2,3], name=123, city="shang")

from abc import ABC, abstractmethod

class A(ABC):  # 继承自ABC使A成为抽象基类
    def ab(self):
        file = 1
        feature = 'feature'
        a = self.abst(file, feature, name='test', software='vscode')
        return a
    
    @abstractmethod
    def abst(self, file, feature, **kwargs):
        pass

class B(A):
    def abst(self, file, feature, **kwargs):
        kwargs_str = '+'.join(f"{k}+{v}" for k, v in kwargs.items())
        return f"{file}+{feature}+{kwargs_str}"

# 正确的使用方法
b = B()
print(b.ab())

