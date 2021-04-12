import cffi


ffi = cffi.FFI()
ffi.cdef("int print();")
C = ffi.dlopen("./libhello.so")


def hello():
    C.print()


if __name__ == "__main__":
    string = "Python is interesting."

# string with encoding 'utf-8'
    hello()
