# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

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
CMAKE_COMMAND = /home/feixue/ProgramFiles/clion-2016.2.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/feixue/ProgramFiles/clion-2016.2.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/feixue/Code/ImageProcess/BoW-SIFT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/feixue/Code/ImageProcess/BoW-SIFT

# Include any dependencies generated for this target.
include CMakeFiles/BoW_SIFT.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/BoW_SIFT.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BoW_SIFT.dir/flags.make

CMakeFiles/BoW_SIFT.dir/main.cpp.o: CMakeFiles/BoW_SIFT.dir/flags.make
CMakeFiles/BoW_SIFT.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/feixue/Code/ImageProcess/BoW-SIFT/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/BoW_SIFT.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/BoW_SIFT.dir/main.cpp.o -c /home/feixue/Code/ImageProcess/BoW-SIFT/main.cpp

CMakeFiles/BoW_SIFT.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BoW_SIFT.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/feixue/Code/ImageProcess/BoW-SIFT/main.cpp > CMakeFiles/BoW_SIFT.dir/main.cpp.i

CMakeFiles/BoW_SIFT.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BoW_SIFT.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/feixue/Code/ImageProcess/BoW-SIFT/main.cpp -o CMakeFiles/BoW_SIFT.dir/main.cpp.s

CMakeFiles/BoW_SIFT.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/BoW_SIFT.dir/main.cpp.o.requires

CMakeFiles/BoW_SIFT.dir/main.cpp.o.provides: CMakeFiles/BoW_SIFT.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/BoW_SIFT.dir/build.make CMakeFiles/BoW_SIFT.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/BoW_SIFT.dir/main.cpp.o.provides

CMakeFiles/BoW_SIFT.dir/main.cpp.o.provides.build: CMakeFiles/BoW_SIFT.dir/main.cpp.o


# Object files for target BoW_SIFT
BoW_SIFT_OBJECTS = \
"CMakeFiles/BoW_SIFT.dir/main.cpp.o"

# External object files for target BoW_SIFT
BoW_SIFT_EXTERNAL_OBJECTS =

BoW_SIFT: CMakeFiles/BoW_SIFT.dir/main.cpp.o
BoW_SIFT: CMakeFiles/BoW_SIFT.dir/build.make
BoW_SIFT: /usr/local/lib/libopencv_viz.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_videostab.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_ts.a
BoW_SIFT: /usr/local/lib/libopencv_superres.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_stitching.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_contrib.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_nonfree.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_ocl.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_gpu.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_photo.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_objdetect.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_legacy.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_video.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_ml.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_calib3d.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_features2d.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_highgui.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_imgproc.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_flann.so.2.4.10
BoW_SIFT: /usr/local/lib/libopencv_core.so.2.4.10
BoW_SIFT: CMakeFiles/BoW_SIFT.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/feixue/Code/ImageProcess/BoW-SIFT/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable BoW_SIFT"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BoW_SIFT.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BoW_SIFT.dir/build: BoW_SIFT

.PHONY : CMakeFiles/BoW_SIFT.dir/build

CMakeFiles/BoW_SIFT.dir/requires: CMakeFiles/BoW_SIFT.dir/main.cpp.o.requires

.PHONY : CMakeFiles/BoW_SIFT.dir/requires

CMakeFiles/BoW_SIFT.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BoW_SIFT.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BoW_SIFT.dir/clean

CMakeFiles/BoW_SIFT.dir/depend:
	cd /home/feixue/Code/ImageProcess/BoW-SIFT && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/feixue/Code/ImageProcess/BoW-SIFT /home/feixue/Code/ImageProcess/BoW-SIFT /home/feixue/Code/ImageProcess/BoW-SIFT /home/feixue/Code/ImageProcess/BoW-SIFT /home/feixue/Code/ImageProcess/BoW-SIFT/CMakeFiles/BoW_SIFT.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BoW_SIFT.dir/depend

