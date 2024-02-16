#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>

#include <stdio.h>
#include <fstream>
#include <iostream>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// The cuda kernel
__global__ void quamsim_kernel( const float *A, 
                                const float *B, 
                                float *C, 
                                int q_bit, 
                                int numElements) {

    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    int j = i/q_bit;

    //A -> Quantum Gate
    //B -> Input Vector; C -> Output Vector
    //q_bit -> target q-bit; size -> total number of elements
    
    if((i<numElements)&&((i+q_bit)<numElements)&&(j%2==0)) {
        C[i]        = A[0]*B[i] + A[1]*B[i+q_bit];
        C[i+q_bit]  = A[2]*B[i] + A[3]*B[i+q_bit];
    }

    // q_bit = q_bit>>1;
    // if (i < numElements) {
    //     // Calculate the index of the adjacent element
    //     bool control_qubit_set = ((i >> q_bit) & 1) == 1;
    //     int j = (control_qubit_set) ? i ^ (1 << q_bit) : i; // Correct calculation  
    //     // // Apply the quantum gate operation
    //     if(j<numElements){
    //     if (control_qubit_set) {
    //         C[i] = A[0] * B[i] + A[1] * B[j]; // Apply with correct index swapping
    //         C[j] = A[2] * B[i] - A[3] * B[j]; // Correct signs for A[2] and A[3]
    //     } else {
    //         C[i] = A[2] * B[i] + A[3] * B[j];
    //         C[j] = A[0] * B[i] + A[1] * B[j];
    //     }
    //     }
    // }
}

int main(int argc, char *argv[]) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Read the inputs from command line
    if (argc != 2) {
        printf("Error: Please provide the input file\n");
        return EXIT_FAILURE;
    }

    const std::string file = argv[1];

    // Open the input file and read the data
    std::ifstream inputFile(file);
    if (!inputFile.is_open()) {
        printf("Error: Could not open the input file\n");
        return EXIT_FAILURE;
    }

    int count = 0;
    float temp;
    while (inputFile >> temp) {
        count++;
    }

    inputFile.close();

    int numElements = count - 5;
    int gate_size = 4*sizeof(float);
    float *A = (float *)malloc(gate_size);
    int arr_size = numElements*sizeof(float);
    float *B = (float *)malloc(arr_size);
    float *C = (float *)malloc(arr_size);
    int q_bit;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    inputFile.open(file);
    // Read the gate matrix elements
    for (int i = 0; i < 4; i++) {
        inputFile >> A[i];
    }
    // Read the state vector elements
    for (int i = 0; i < numElements; i++) {
        inputFile >> B[i]; 
    }
    // Read the gate target qubit index
    inputFile >> q_bit;
    q_bit = 1<<q_bit;
    inputFile.close();
    


    // Allocate/move data using cudaMalloc and cudaMemCpy
    // float *d_A, *d_B, *d_C;
    // cudaMalloc(&d_A, gate_size);
    // cudaMalloc(&d_B, arr_size);
    // cudaMalloc(&d_C, arr_size);
    // cudaMemcpy(d_A, A, gate_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, B, arr_size, cudaMemcpyHostToDevice);

    // Allocate the device input vector A (2X2 Qunatum Gate)
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, gate_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B (Qunatum State Vector)
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, arr_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate the device input vector C (Qunatum State Vector - output)
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, arr_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, A, gate_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, B, arr_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    // Launch the kernel and take timestamps before and after
    cudaEventRecord(start);
    quamsim_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, q_bit, numElements);
    cudaEventRecord(stop);
    err = cudaGetLastError();


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(C, d_C, arr_size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize(); // Wait for kernel to finish

    // // Copy the output back to host memory
    // cudaMemcpy(C, d_C, arr_size, cudaMemcpyDeviceToHost);
    // Print the output
    printf("Quantum Gate Matrix A:\n");
    for(int i=0; i<4; i++){
    printf("%.3f ", A[i]);
    if ((i+1) % 2 == 0) printf("\n");
    }

    printf("Input Vector B:\n");
    for(int i=0; i<numElements; i++){
    printf("%.3f \n", B[i]);
    }

    printf("q_bit = %d\n", q_bit);

    printf("Output Vector C:\n");
    for(int i=0; i<numElements; i++){
    printf("%.3f \n", C[i]);
    }


    // Extract the timing information
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f\n", milliseconds);
    // printf("%f\n", cudaEventElapsedTime);


    // Clean up the memory
    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(A);
    free(B);
    free(C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}




/**
/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

*/
