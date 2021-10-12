# CC=g++
CFLAGS=-L/Users/Saquib/opt/anaconda3/envs/snn/bin/ -framework Python
CFLAGS2=-I/Users/Saquib/opt/anaconda3/envs/snn/include/python3.8 -I/Users/Saquib/opt/anaconda3/envs/snn/include/python3.8 -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/Users/Saquib/opt/anaconda3/envs/snn/include -arch x86_64 -I/Users/Saquib/opt/anaconda3/envs/snn/include -arch x86_64
LDFLAGS=-L/Users/Saquib/opt/anaconda3/envs/snn/lib/python3.8/config-3.8-darwin -ldl -framework CoreFoundation

CC=clang++
CFLAGSBIND=-std=c++11 -shared -undefined dynamic_lookup -I./pybind11/include/ `python3.8 -m pybind11 --includes` -L/Users/Saquib/opt/anaconda3/envs/snn/bin/ -framework Python
LDFLAGSBIND=`python3.8-config -m pybind11 --ldflags`

default:
	# $(CC) $(CFLAGS) -o prog simulation.cpp `python-config --include`
	$(CC) $(CFLAGS) $(CFLAGS2) $(LDFLAGS) -o prog simulation.cpp

bind:
	clang++ -std=c++11 -shared -undefined dynamic_lookup -I./pybind11/include/ `python3.8 -m pybind11 --includes` -L/Users/Saquib/opt/anaconda3/envs/snn/bin/ -framework Python simulation.cpp -o prog `python3.8-config -m pybind11 --ldflags`

clean:
	rm prog