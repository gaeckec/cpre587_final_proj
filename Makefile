.PHONY: clean depend
.SUFFIXES: .o
.SECONDARY:

# Recursive Wildcard (https://stackoverflow.com/questions/2483182/recursive-wildcards-in-gnu-make/18258352#18258352)
rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))
GCCVERSIONLTQ9 := $(shell expr `gcc -dumpversion | cut -f1 -d.` \< 9)

DEBUG ?= true
BDIR = build
BIN = $(BDIR)
SDIR = src
EXE = $(BIN)/ml

# ifeq ($(OS), Windows_NT) # Windows
# 	CC_Linux = 
# 	CC_WIN = cl
# 	WIN_SETUP = vcvarsall.bat amd64
#	EXECUTE_EXT = .exe
# else # Linux
	CC_LINUX = g++ 
	CC_WIN = x86_64-w64-mingw32-g++ -static-libgcc -static-libstdc++ -fstack-protector
	CC_ALL = -lstdc++ -Wall -std=c++17
	CC_DEBUG = -g
	CC_OPT_FLAGS = -O3 -march=native -fno-tree-pre
	CC_FLAGS = $(CC_ALL) $(DEPEND_FLAGS) $(if $(filter $(DEBUG), true), $(CC_DEBUG), $(CC_OPT_FLAGS))
	
#	Check if GCC is version 9 or greater, and if not, include an additional flag for the filesystem lib
	ifeq "$(GCCVERSIONLTQ9)" "1"
    	CC_FLAGS_END += -lstdc++fs
	endif
# endif

INC = -Iinc
SOURCE_FILES = $(call rwildcard, $(SDIR), *.cpp)	# Find all source files
DEPEND_FILES = $(call rwildcard, $(BDIR), *.d)

_OBJS = $(patsubst %.cpp, %.o, $(SOURCE_FILES))		# Calculate names of object files by replacing .c and .cpp with .o
OBJS = $(patsubst $(SDIR)/%, $(BDIR)/%, $(_OBJS))	# Create paths for those names by appending the build dir

-include $(DEPEND_FILES)

# Debug Prints
# $(warning SOURCE_FILES: $(SOURCE_FILES))
# $(warning HEADER_FILES: $(HEADER_FILES))
# $(warning _OBJS: $(_OBJS))
# $(warning OBJS: $(OBJS))
# $(warning EXE: $(EXE))
# $(warning DEBUG: $(DEBUG))


# Make Rules
all: clean dir_struct build

# Build compiler from source
build: dir_struct $(EXE)

rebuild: clean build

# Generate object files
$(BIN)/%: $(OBJS)
	$(CC_LINUX) $(CC_FLAGS) $(OBJS) -o $@ $(CC_FLAGS_END)

$(BDIR)/%.o: $(SDIR)/%.cpp
	mkdir -p $(dir $@)
	$(CC_LINUX) $(CC_FLAGS) -c $(INC) -o $@ $< $(CFLAGS)

# Clean build files
clean:
	@rm -rf $(BDIR)/

# Generate the build folder structure
dir_struct:
	@mkdir -p $(BDIR)/ $(BDDIR)/
