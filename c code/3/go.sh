set -e

[ -z "$PYTHON" ] && PYTHON="python"

g++ -o ./hello ./libhello.cpp
echo "=> now run the native one"
./hello

g++ -o ./libhello.so ./test.cpp -fPIC -shared
echo "=> now run the $PYTHON bound one"
python hello.py
