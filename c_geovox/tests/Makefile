INC_DIR    = ../include
LIB_DIR    = ../lib

TARGET     = test

SRC        = $(wildcard *.cpp)
OBJ        = $(SRC:.cpp=.o)

CC         = g++ -fopenmp
CFLAGS     = -O2 -Wall
LDFLAGS    = -L$(LIB_DIR)
LDLIBS     = -lcgeovox


.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJ) $(LIB_DIR)/libcgeovox.a
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)


.cpp.o:
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

.PHONY: clean
clean:
	rm -rv $(TARGET) $(OBJ)

.PHONY: print
print:
	@echo SOURCES:= $(SRC)
	@echo OBJECTS:= $(OBJ)