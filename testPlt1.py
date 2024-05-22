

from testPlt import A


class B(A):
    def abst(self, file, feature, **kwargs):
        return file+feature+kwargs
    
a = B()
print(a.ab())