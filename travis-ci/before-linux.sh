export CMAKE_CXX_FLAGS="-DAFFINITY"
if [[ -v CUDA ]]; then
	cmake .. -DFIRESTARTER_BUILD_CUDA=ON ;
else
	cmake .. ;
fi
