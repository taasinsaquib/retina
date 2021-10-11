CC=g++
CFLAGS=-framework Python

default:
	$(CC) $(CFLAGS) -o prog simulation.cpp `python-config --include`

clean:
	rm prog