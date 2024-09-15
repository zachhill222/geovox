INC_DIR    = ../include
LIB_DIR    = ../lib

TARGET     = test

SRC        = $(wildcard *.cpp)
OBJ        = $(SRC:.cpp=.o)

CC         = g++
CPPFLAGS   = -I$(INC_DIR)
CFLAGS     = -O2 -Wall
LDFLAGS    = -L$(LIB_DIR)
LDLIBS     = -lcgeovox


.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJ) $(LIB_DIR)/libcgeovox.a
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rv $(TARGET) $(OBJ)

.PHONY: print
print:
	@echo SOURCES:= $(SRC)
	@echo OBJECTS:= $(OBJ)