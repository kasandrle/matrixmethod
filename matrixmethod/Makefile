OPTIMIZE=-flto=4 -fno-fat-lto-objects --param ggc-min-heapsize=6291456

OPTIMIZE += -fbranch-target-load-optimize	# Perform branch target register load optimization before prologue / epilogue threading. 
OPTIMIZE += -fgcse-las				# When -fgcse-las is enabled, the global common subexpression elimination pass eliminates redundant loads that come after stores to the same memory location (both partial and full redundancies). 
OPTIMIZE += -fgcse-sm				# When -fgcse-sm is enabled, a store motion pass is run after global common subexpression elimination.
#Not working with LTO 		OPTIMIZE += -fipa-pta				# Perform interprocedural pointer analysis and interprocedural modification and reference analysis.
OPTIMIZE += -fira-loop-pressure			# Use IRA to evaluate register pressure in loops for decisions to move loop invariants.
OPTIMIZE += -fgraphite-identity			# Enable the identity transformation for graphite.
OPTIMIZE += -floop-block			# Perform loop blocking transformations on loops.
OPTIMIZE += -floop-interchange			# Perform loop interchange transformations on loops.
OPTIMIZE += -floop-nest-optimize		# Enable the ISL based loop nest optimizer.
OPTIMIZE += -floop-strip-mine			# Perform loop strip mining transformations on loops.
# 		OPTIMIZE += -floop-unroll-and-jam		# Enable unroll and jam for the ISL based loop nest optimizer. 
OPTIMIZE += -fmodulo-sched 			# Perform swing modulo scheduling immediately before the first scheduling pass. 
OPTIMIZE += -fmodulo-sched-allow-regmoves 	# Perform more aggressive SMS-based modulo scheduling with register moves allowed. 
OPTIMIZE += -fomit-frame-pointer		# Don't keep the frame pointer in a register for functions that don't need one.
OPTIMIZE += -fsched-pressure			# Enable register pressure sensitive insn scheduling before register allocation. 
OPTIMIZE += -fsched-spec-load			# Allow speculative motion of some load instructions. 
OPTIMIZE += -fsched2-use-superblocks		# When scheduling after register allocation, use superblock scheduling.
OPTIMIZE += -ftree-loop-distribution		# Perform loop distribution.
OPTIMIZE += -ftree-vectorize			# Perform vectorization on trees.

OPTIMIZE += -fcx-fortran-rules			# Complex multiplication and division follow Fortran rules.
OPTIMIZE += -fcx-limited-range          # Complex multiplication and division assume no inf/NaNs.
OPTIMIZE += -fsched-spec-load-dangerous 	# Allow speculative motion of more load instructions.
OPTIMIZE += -funsafe-loop-optimizations 	# This option tells the loop optimizer to assume that loop indices do not overflow, and that loops with nontrivial exit condition are not infinite. 

OPTIMIZE += --param max-vartrack-expr-depth=120


# Potentially loose precicision
OPTIMIZE += -freciprocal-math		# Allow the reciprocal of a value to be used instead of dividing by the value if this enables optimizations.

# Potentially VERY dangerous
OPTIMIZE += -fmerge-all-constants	# Attempt to merge identical constants and identical variables. 
OPTIMIZE += -fno-math-errno 		# Do not set errno after calling math functions that are executed with a single instruction
OPTIMIZE += -funsafe-math-optimizations # Allow optimizations for floating-point arithmetic that (a) assume that arguments and results are valid and (b) may violate IEEE or ANSI standards.
OPTIMIZE += -ffinite-math-only		# Allow optimizations for floating-point arithmetic that assume that arguments and results are not NaNs or +-Infs. 
OPTIMIZE += -fno-signed-zeros		# Allow optimizations for floating-point arithmetic that ignore the signedness of zero.
OPTIMIZE += -fno-trapping-math		# Compile code assuming that floating-point operations cannot generate user-visible traps. These traps include division by zero, overflow, underflow, inexact result and invalid operation.


# GRAPHITE
OPTIMIZE += -floop-interchange		#Perform loop interchange transformations on loops.
OPTIMIZE += -floop-strip-mine		#Perform loop strip mining transformations on loops.
OPTIMIZE += -floop-block		#Perform loop blocking transformations on loops.
OPTIMIZE += -fgraphite-identity
OPTIMIZE += -floop-nest-optimize

GCC5 = /var/tmp/mpfluege-s/gcc/bin/g++
GCC5FLAGS = -I /var/tmp/mpfluege-s/gcc/include -L /var/tmp/mpfluege-s/gcc/lib64

IPPFLAGS = -I /var/tmp/mpfluege-s/intel/ipp/include -L /var/tmp/mpfluege-s/intel/ipp/lib

matrixmethod: matrixmethod.cpp
	g++ -Wall -std=c++11 -O3 -march=native matrixmethod.cpp -o matrixmethod
	g++ -Wall -std=c++11 -Ofast -march=native matrixmethod.cpp -o matrixmethod_ofast
	g++ -Wall -std=c++11 -Ofast -march=native ${OPTIMIZE} matrixmethod.cpp -o matrixmethod_opt
	#g++ -Wall -std=c++11 -Ofast -march=native ${OPTIMIZE} matrixmethod.cpp -o matrixmethod_openmp
	g++ -Wall -std=c++11 -Ofast -march=native -g ${OPTIMIZE} matrixmethod.cpp -o matrixmethod_debug
	${GCC5} ${GCC5FLAGS} -Wall -std=c++11 -Ofast -march=native ${OPTIMIZE} -ftree-vectorizer-verbose=2 -fopt-info-vec-missed matrixmethod.cpp -o matrixmethod_gcc5_opt

matrixmethod_eigen: matrixmethod_eigen.cpp libmatrixmethod_eigen.cpp
	g++ -Wall -std=c++11 -I . -O3 -march=native -mtune=native matrixmethod_eigen.cpp -o matrixmethod_eigen
	g++ -Wall -std=c++11 -I . -Ofast -march=native -mtune=native matrixmethod_eigen.cpp -o matrixmethod_eigen_ofast


matrixmethod_ipp: matrixmethod_ipp.cpp
	${GCC5} ${GCC5FLAGS} -Wall -std=c++11 -Ofast -march=native ${OPTIMIZE} ${IPPFLAGS} -ftree-vectorizer-verbose=2 matrixmethod_ipp.cpp -o matrixmethod_ipp_opt

clean:
	rm -f matrixmethod
	rm -f matrixmethod_native
	rm -f matrixmethod_clang
	rm -f matrixmethod_ofast
	rm -f matrixmethod_eigen
	rm -f matrixmethod_eigen_native
	rm -f matrixmethod_eigen_clang
	rm -f matrixmethod_eigen_ofast
	rm -f matrixmethod_opt
	rm -f matrixmethod_debug
	rm -f matrixmethod_openmp
	rm -f matrixmethod_gcc5_opt
