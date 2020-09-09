source travis-ci/install-cmake-3-13.sh
if [[ -v CUDA ]]; then
		source travis-ci/install-cuda-xenial.sh ;
fi
