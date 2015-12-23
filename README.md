#GPU Graph Applications in CUDA 
This suite contains Graph Applications used in the following IISWC paper:

Q. Xu, H. Jeon, M. Annavaram (University of Southern California), Graph processing on GPUs: Where are the bottlenecks?, In Proceedings of the IEEE International Symposium on Workload Characterization (IISWC), Oct. 2014. 

BFS, MST, SP, SSSP which are from existing benchmark suites (Rodinia, Lonestar) are not listed here.

Please cite if your use of this repository results in a publication. 

#System Requirements
1. Linux® (tested with CentOS release 5.9)
2. CUDA 4.0 & CUDA 5.5 
3. NVIDIA's CUDA development libraries and tools
4. Cusp v0.4.0
5. Boost development libraries
6. Intel's Threading Building Blocks library

#Installation and Building
1. Put the Graph Application folder under NVIDIA_GPU_Computing_SDK4/C/ directory
2. Add the following lines in NVIDIA_GPU_Computing_SDK4/C/common/common.mk 

ifeq ($(USEBOOST),1)
  LIB += -lboost_iostreams-mt -lboost_program_options-mt
endif

ifeq ($(USETBB),1)
  LIB += -ltbb
endif

3. Step into each folder and type make. 

#Running
1. Run the applications using the sample commands in the run file.
2. To run GCL, MIS, PR on GPGPUSim, you may want to patch the GPGPUSim with CUDA5.5 capability.
   A detailed instruction is provided here: http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu/gpgpusim 
   Other benchmarks work well with CUDA4.0. 


