class A(object):
    def __init__(self, a):
        self.a = a
        return

    def load(self):
        print("A " + self.a)


class B(object):
    def __init__(self, a):
        self.a = a
        return

    def load(self):
        print("B " + self.a)


class C(A, B):
    def __init__(self, a):
        super().__init__(a)
        return

    def load(self):
        super().load()
        return

if __name__ == "__main__":
    ca = C("a")
    ca.load()
    cb = C("q")
    cb.load()