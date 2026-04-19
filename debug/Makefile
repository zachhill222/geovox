#set up the include directories
INC_DIR = -I../include -I../libs/GeometryUtility/include -I/usr/include/eigen3

#get source/target names of all files in the specified directory
DIR    ?= .
SRC     = $(wildcard $(DIR)/*.cpp)
TARGETS = $(SRC:.cpp=)

#default build (debug)
CXX        =  g++ -std=c++20 -pg
OPT        = -O0
PAR        =
DEBUG      = -pg -g3 -Wall

#redefine for testing optimizations
CXX        =  g++ -std=c++20
OPT        = -O3
PAR        = -fopenmp
DEBUG      = -DNDEBUG

#final compiler command
CC = $(CXX) $(OPT) $(PAR) $(DEBUG)

all: $(TARGETS)

$(DIR)/%: $(DIR)/%.cpp FORCE
	$(CC) $(INC_DIR) $< -o $@

FORCE: ;


#TODO: the clean up commands need to be changed
.PHONY: clean
clean: remove_tests
	rm -f $(TARGETS) *.o


.PHONY: remove_tests
remove_tests:
	rm -f gmon.out
	rm -rf *.er
	rm -f *.vtk
	rm -f ./outfiles/*
	rm -f massif.*

.PHONY: print
print:
	@echo SOURCES:= $(SRC)
	@echo TARGETS:= $(TARGETS)
	@echo INC_DIR:= $(INC_DIR)