# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.7.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.7.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp

# Include any dependencies generated for this target.
include CMakeFiles/Iso.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Iso.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Iso.dir/flags.make

CMakeFiles/Iso.dir/main.cpp.o: CMakeFiles/Iso.dir/flags.make
CMakeFiles/Iso.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Iso.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Iso.dir/main.cpp.o -c /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp/main.cpp

CMakeFiles/Iso.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Iso.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp/main.cpp > CMakeFiles/Iso.dir/main.cpp.i

CMakeFiles/Iso.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Iso.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp/main.cpp -o CMakeFiles/Iso.dir/main.cpp.s

CMakeFiles/Iso.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/Iso.dir/main.cpp.o.requires

CMakeFiles/Iso.dir/main.cpp.o.provides: CMakeFiles/Iso.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Iso.dir/build.make CMakeFiles/Iso.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/Iso.dir/main.cpp.o.provides

CMakeFiles/Iso.dir/main.cpp.o.provides.build: CMakeFiles/Iso.dir/main.cpp.o


# Object files for target Iso
Iso_OBJECTS = \
"CMakeFiles/Iso.dir/main.cpp.o"

# External object files for target Iso
Iso_EXTERNAL_OBJECTS =

Iso: CMakeFiles/Iso.dir/main.cpp.o
Iso: CMakeFiles/Iso.dir/build.make
Iso: CMakeFiles/Iso.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Iso"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Iso.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Iso.dir/build: Iso

.PHONY : CMakeFiles/Iso.dir/build

CMakeFiles/Iso.dir/requires: CMakeFiles/Iso.dir/main.cpp.o.requires

.PHONY : CMakeFiles/Iso.dir/requires

CMakeFiles/Iso.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Iso.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Iso.dir/clean

CMakeFiles/Iso.dir/depend:
	cd /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp /Users/achm/Documents/github/kaggle-allstate/ISO_REG_cpp/CMakeFiles/Iso.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Iso.dir/depend

