INC_DIR   := include
BUILD_DIR := build

SOURCES   := $(wildcard src/*.cpp)
OBJECTS   := $(SOURCES:.cpp=.o)


CC        := g++
CFLAGS    := -O2 -Wall
LDFLAGS   :=
LIBS      := 

TARGET    := $(BUILD_DIR)/libcgeovox.a


# build library
$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	ar rcs $@ $^
	ranlib $@
	cp -rfu $(INC_DIR) $(BUILD_DIR)/include

# build objects
.cpp.o:
	$(CC) $(CFLAGS) $(LDFLAGS) -I$(INC_DIR) -c $< -o $@ 

# create directories for build and objects if they don't exist
$(BUILD_DIR):
	mkdir -p $@

# clean directories
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm *.o

.PHONY: print
print:
	@echo SOURCES:= $(SOURCES)
	@echo OBJECTS:= $(OBJECTS)