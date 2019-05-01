#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__device__ int hashFunc(int key, int offset = 0 )
{
    offset *= 2;
    uint64_t a = primeList[MAX_HASH_PARAM-offset];
    uint64_t b = primeList[MAX_HASH_PARAM-offset-1];
    return (a*key + b) % 4294967291U;
}

__global__ void cuckooInsert( int* keys, int* values, int numKeys, Bucket *table, int capacity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numKeys) {
        int key = keys[idx];
        int value = values[idx];
        Bucket newValue = (static_cast<Bucket>(key) << 1*sizeof(Value)) | value;

        uint32_t idx[5];
        idx[0] = hashFunc(key, 0) % capacity;

        for (int i = 0; i < MAX_VER; ++i) {
            
            newValue = atomicExch(reinterpret_cast<unsigned long long*>(&table[idx[0]]), newValue);
            if (newValue & 0xff00 == KEY_INVALID)
                return;

            // Otherwise find a new location for the displaced item
            idx[1] = hashFunc( key, 0 ) % capacity;
            idx[2] = hashFunc( key, 1 ) % capacity;
            idx[3] = hashFunc( key, 2 ) % capacity;
            idx[4] = hashFunc( key, 3 ) % capacity;

            if( idx[0] == idx[1] ) idx[0] = idx[2];
            else if( idx[0] == idx[2] ) idx[0] = idx[3];
            else if( idx[0] == idx[3] ) idx[0] = idx[4];
            else idx[0] = idx[1];
        }

        printf("tid %d: Insert (%u,%u) failed\n", idx, key, value);
    }
    return;
}

__global__ void cuckooGet( int* keys, int* values, int numKeys, Bucket *table, int capacity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numKeys) {
        int key = keys[idx];

        // Compute all possible locations
        uint32_t idx[4];
        idx[0] = hashFunc(key, 0 );
        idx[1] = hashFunc(key, 1 );
        idx[2] = hashFunc(key, 2 );
        idx[3] = hashFunc(key, 3 );

        Bucket entry;
        for( int i = 0; i < 4; ++i )
        {
            entry = table[idx[i] % ts->capacity];
            Key k = entry >> 1*sizeof(Value);
            if( k == key ) {
                values[idx] = entry & 0xff;
                return;
            }
            if( k == NULL_KEY )
                break;
        }

        // Should never fail except for invalid keys
        printf( "Query for %u failed\n", key );
    }

    return;
}



/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {

    capacity = size;
    // free slots have <key, value> equal to 0
    cudaMalloc((void **) &table, capacity * sizeof(Bucket));
    cudaMemset(table, KEY_INVALID, capacity * sizeof(Bucket));
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

    Bucket *newTable = NULL;
    // free slots have <key, value> equal to 0
    cudaMalloc((void **) &newTable, numBucketsReshape * sizeof(Bucket));
    cudaMemset(newTable, KEY_INVALID, numBucketsReshape * sizeof(Bucket));
    if (table != NULL) {
        cudaMemcpy(newTable, table, capacity * sizeof(Bucket), cudaMemcpyDeviceToDevice);
        cudaFree(table);
        capacity = numBucketsReshape;
    }
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {

    currentSize += numKeys;
    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));
    cudaMemcpy(deviceKeys, keys, numKeys, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys, cudaMemcpyHostToDevice);

    const int block_size = 64;
    int blocks_no = numKeys / block_size;

    if (numKeys % block_size)
        ++blocks_no;

    // Launch the kernel
    cuckooInsert <<< blocks_no, block_size >>> (deviceKeys, deviceValues, numKeys, table, capacity);

    cudaDeviceSynchronize();
    cudaFree(deviceKeys);
    cudaFree(deviceValues);
    return false;
}

/* GET BATCH
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));
    cudaMemcpy(deviceKeys, keys, numKeys, cudaMemcpyHostToDevice);

    const int block_size = 64;
    int blocks_no = numKeys / block_size;

    if (numKeys % block_size)
        ++blocks_no;
    cuckooGet <<< blocks_no, block_size >>> (deviceKeys, deviceValues, numKeys, table, capacity);

    int *hostValues = 0;

    hostValues = (int*) malloc(numKeys * sizeof(int));
    cudaMemcpy(hostValues, deviceValues, numKeys, cudaMemcpyDeviceToHost);
    return hostValues;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
    return currentSize / 1.0 * capacity; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
