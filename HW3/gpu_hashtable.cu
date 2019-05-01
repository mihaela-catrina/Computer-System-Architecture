#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__global__ void cuckooInsert( int* keys, int* values )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int key = keys[idx];
    int value = values[idx];
    printf( "tid %d: Insert (%u,%u) failed\n", idx, key, value );
    return;
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
    capacity = size;
    // free slots have <key, value> equal to 0
    cudaMalloc((void **) &table, capacity * sizeof(Bucket));
    cudaMemset(table, 0, capacity * sizeof(Bucket));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(table);
    currentSize = 0;
    capacity = 0;
    table = NULL;
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {

    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));
    cudaMemcpy(deviceKeys, keys, numKeys, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys, cudaMemcpyHostToDevice);

    const int block_size = 64;
    int blocks_no = numKeys / block_size;

    if (numKeys % block_size)
        ++blocks_no;

    // Launch the kernel
    cuckooInsert <<< blocks_no, block_size >>> (deviceKeys, deviceValues);

    cudaDeviceSynchronize();
    return false;
}

/* GET BATCH
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
    return NULL;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
    return 0.f; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
