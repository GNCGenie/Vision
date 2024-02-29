# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/Inspecity/Codes/CameraCalibration

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/Inspecity/Codes/CameraCalibration/build

# Include any dependencies generated for this target.
include CMakeFiles/pose_estimation_book2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pose_estimation_book2.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pose_estimation_book2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pose_estimation_book2.dir/flags.make

CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.o: CMakeFiles/pose_estimation_book2.dir/flags.make
CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.o: /home/Inspecity/Codes/CameraCalibration/src/pose_estimation_book2.cpp
CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.o: CMakeFiles/pose_estimation_book2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/Inspecity/Codes/CameraCalibration/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.o -MF CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.o.d -o CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.o -c /home/Inspecity/Codes/CameraCalibration/src/pose_estimation_book2.cpp

CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/Inspecity/Codes/CameraCalibration/src/pose_estimation_book2.cpp > CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.i

CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/Inspecity/Codes/CameraCalibration/src/pose_estimation_book2.cpp -o CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.s

# Object files for target pose_estimation_book2
pose_estimation_book2_OBJECTS = \
"CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.o"

# External object files for target pose_estimation_book2
pose_estimation_book2_EXTERNAL_OBJECTS =

bin/pose_estimation_book2: CMakeFiles/pose_estimation_book2.dir/src/pose_estimation_book2.cpp.o
bin/pose_estimation_book2: CMakeFiles/pose_estimation_book2.dir/build.make
bin/pose_estimation_book2: /usr/lib/libopencv_gapi.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_stitching.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_alphamat.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_aruco.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_bgsegm.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_bioinspired.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_ccalib.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_cvv.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_dnn_objdetect.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_dnn_superres.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_dpm.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_face.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_freetype.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_fuzzy.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_hdf.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_hfs.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_img_hash.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_intensity_transform.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_line_descriptor.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_mcc.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_quality.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_rapid.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_reg.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_rgbd.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_saliency.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_stereo.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_structured_light.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_superres.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_surface_matching.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_tracking.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_videostab.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_viz.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_wechat_qrcode.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_xfeatures2d.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_xobjdetect.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_xphoto.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_shape.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_highgui.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_datasets.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_plot.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_text.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_ml.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_phase_unwrapping.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_optflow.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_ximgproc.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_video.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_videoio.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_imgcodecs.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_objdetect.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_calib3d.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_dnn.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_features2d.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_flann.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_photo.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_imgproc.so.4.9.0
bin/pose_estimation_book2: /usr/lib/libopencv_core.so.4.9.0
bin/pose_estimation_book2: CMakeFiles/pose_estimation_book2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/Inspecity/Codes/CameraCalibration/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/pose_estimation_book2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pose_estimation_book2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pose_estimation_book2.dir/build: bin/pose_estimation_book2
.PHONY : CMakeFiles/pose_estimation_book2.dir/build

CMakeFiles/pose_estimation_book2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pose_estimation_book2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pose_estimation_book2.dir/clean

CMakeFiles/pose_estimation_book2.dir/depend:
	cd /home/Inspecity/Codes/CameraCalibration/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/Inspecity/Codes/CameraCalibration /home/Inspecity/Codes/CameraCalibration /home/Inspecity/Codes/CameraCalibration/build /home/Inspecity/Codes/CameraCalibration/build /home/Inspecity/Codes/CameraCalibration/build/CMakeFiles/pose_estimation_book2.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/pose_estimation_book2.dir/depend
