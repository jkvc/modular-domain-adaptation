class loader:
    def __iter__(self):
        self.a = 0
        return self

    def __next__(self):
        if self.a < 5:
            self.a += 1
            return self.a
        else:
            raise StopIteration


l = loader()
print(list(l))
print(list(l))
