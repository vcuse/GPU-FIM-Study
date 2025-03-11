todo: cudaItemsetTemplate
cudaItemsetTemplate: cudaItemsetTemplate.cu
	nvcc --expt-extended-lambda --ptxas-options=-v -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -IcuCollections/include -Icccl/libcudacxx/include -Icccl/cub -Icccl/thrust -Iwarpcore/include -Ihpc_helpers/include cudaItemsetTemplate.cu -o cudaItemsetTemplate

clean:
	rm cudaItemsetTemplate

