export CMAKE_CXX_FLAGS="-DAFFINITY"
if [[ -v CUDA ]]; then
	cmake .. -DBUILD_CUDA=1 ;
else
	cmake .. ;
fi
