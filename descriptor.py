class D:
    def __get__(self, instance, owner):
        print("called")


class A:
    pass


a = A()
a.x = D()

print(a.x)  # just returns D object


class B:
    # x = D()
    x = 10
    # def __init__(self, x=None) -> None:
    #     self.x = x


b = B()
b.x = D()

print(b.x)  # just returns D object


class C:
    # x = D()
    # x = 10
    def x(self):
        print("original descriptor")


c = C()
c.x = D()

print(c.x)  # just returns D object
