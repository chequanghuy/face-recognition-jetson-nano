import ctypes
import pathlib

if __name__ == "__main__":
    # Load the shared library into ctypes
    libname = pathlib.Path().absolute() / "test.so"
    print(libname)
    c_lib = ctypes.CDLL(libname)