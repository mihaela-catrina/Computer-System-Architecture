#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__device__ int hashFunc(int *hashConstants, int key, int capacity, int offset = 0) {
    // offset *= 2;
    // printf("Offset = %d\n", offset);
    // int a = hashConstants[19-offset];
    // int b = hashConstants[19-offset-1];
    // printf("a = %d, b = %d\n", a, b);
    // return ((a*key + b) % 4294967291U) % capacity;
    if (offset == 0) return hash1(key, capacity);
    if (offset == 1) return hash2(key, capacity);
    if (offset == 2) return hash3(key, capacity);
}

__global__ void cuckooInsert(int *keys, int *values, int numKeys, Bucket *table, int capacity, bool success) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numKeys) {
        int key = keys[tid];
        int value = values[tid];
        Bucket newValue = (((static_cast<Bucket>(key) << 32)) | (static_cast<Bucket>(value)));

        int idx[4];
        idx[0] = hashFunc(hashConstants, key, capacity, 0);
        for (int i = 0; i < 7 * log2f(numKeys); ++i) {
            newValue = atomicExch(&table[idx[0]], newValue);
            if ((newValue & 0xffffffff00000000) >> 32 == KEY_INVALID) {
                return;
            }
            key = (newValue & 0xffffffff00000000) >> 32;
            value = (newValue & 0x00000000ffffffff);
            // Otherwise find a new location for the displaced item
            int last_loc = idx[0];
            idx[0] = hashFunc(hashConstants, key, capacity, 0);
            idx[1] = hashFunc(hashConstants, key, capacity, 1);
            idx[2] = hashFunc(hashConstants, key, capacity, 2);
            //idx[3] = hashFunc(hashConstants, key, capacity, 3);

            for (int i = 1; i >= 0; --i)
                idx[0] = (last_loc == idx[i] ? idx[i + 1] : idx[0]);

        }
        /*
      if ((newValue & 0xffffffff00000000) >> 32 != KEY_INVALID)
  {	key = (newValue & 0xffffffff00000000)>>32;
      idx[0] = hashFunction(_hashConstants[0], key, capacity);
      auto slot = (unsigned long long int*)(table + (current_size + idx[0]));
      auto replaced = atomicCAS(slot, 0, newValue);
      if (replaced != 0) return;
      } */

        printf("tid %d: Insert (%u,%u) failed\n", tid, key, value);
        success = false;
    }
    return;
}

__global__ void cuckooGet(int *keys, int *values, int numKeys, Bucket *table, int capacity, bool success) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numKeys) {
        int key = keys[tid];

        // Compute all possible locations
        int idx[4];
        idx[0] = hashFunc(hashConstants, key, capacity, 0);
        idx[1] = hashFunc(hashConstants, key, capacity, 1);
        idx[2] = hashFunc(hashConstants, key, capacity, 2);
        // idx[3] = hashFunc(hashConstants, key, capacity, 3 );

        Bucket entry;
        for (int i = 0; i < 3; ++i) {
            entry = static_cast<Bucket> (table[idx[i] % capacity]);
            Key k = (entry & 0xffffffff00000000) >> 32;
            if (k == key) {
                int val = (entry & 0x00000000ffffffff);
                values[tid] = val;
                return;
            }
            if (k == KEY_INVALID)
                break;
        }

        // Should never fail except for invalid keys
        printf("Query for %u failed\n", key);
        success = false;
    }

    return;
}


/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {

    capacity = 0;//size;
    currentSize = 0;
    // free slots have <key, value> equal to 0
//    cudaMalloc((void **) &table, capacity * sizeof(Bucket));
//    cudaMemset(table, KEY_INVALID, capacity * sizeof(Bucket));

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
    capacity = numBucketsReshape;
    table = newTable;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {

    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));
    oldKeys = malloc(currentSize * sizeof(int));
    oldValues = malloc(currentSize * sizeof(int));
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    Bucket *tableValues = 0;

    tableValues = (Bucket *) malloc(capacity * sizeof(Bucket));
    cudaMemcpy(tableValues, table, capacity * sizeof(Bucket), cudaMemcpyDeviceToHost);
    int index = 0;
    for (int i = 0; i < capacity; i++) {
        if(tableValues[i] != KEY_INVALID) {
            oldValues[index] = (tableValues[i] & 0x00000000ffffffff);
            oldKeys[index] = (tableValues[i] & 0xffffffff00000000) >> 32;
        }
    }
    oldSize = currentSize;
    free(tableValues);

    currentSize += numKeys;


    printf("Insertttt\n");
    for (int i = 0; i < numKeys; i++) {
        printf("<key, value> : <%d, %d>\n", keys[i], values[i]);
    }
    const int block_size = 64;
    int blocks_no = numKeys / block_size;

    if (numKeys % block_size)
        ++blocks_no;
    bool success = true;
    // Launch the kernel
    cuckooInsert << < blocks_no, block_size >> > (deviceKeys, deviceValues, numKeys, table, capacity, &success);
    cudaDeviceSynchronize();

    if (!success) {
        rehash(keys, values, numKeys);
        return false;
    }

    Bucket *hostValues = 0;

    hostValues = (Bucket *) malloc(capacity * sizeof(Bucket));
    cudaMemcpy(hostValues, table, capacity * sizeof(Bucket), cudaMemcpyDeviceToHost);
    printf("After insert in table:\n");
    for (int i = 0; i < capacity; i++) {
        printf("Key = %d -> ", (hostValues[i] & 0xffffffff00000000) >> 32);
        printf("Value = %d\n", (hostValues[i] & 0x00000000ffffffff));
    }

    cudaFree(deviceKeys);
    cudaFree(deviceValues);
    free(oldKeys);
    free(oldValues);
    deviceKeys = 0;
    deviceValues = 0;
    oldKeys = 0;
    oldValues = 0;
    cudaDeviceSynchronize();
    return false;
}

void GpuHahTable::rehash(int *keys, int*values, int numKeys) {
    Bucket *newTable = NULL;
    BUcket *aux = 0;
    // free slots have <key, value> equal to 0
    cudaMalloc((void **) &newTable, 2 * capacity * sizeof(Bucket));
    cudaMemset(newTable, KEY_INVALID, 2 * capacity * sizeof(Bucket));
    capacity *= 2;
    aux = table;
    table = newTable;
    cudaFree(aux);

    insertBatch(oldKeys, oldValues, oldSize);
    insertBatch(keys, values, numKeys);

}

/* GET BATCH
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    bool success = true;
    const int block_size = 64;
    int blocks_no = numKeys / block_size;

    if (numKeys % block_size)
        ++blocks_no;
    cuckooGet << < blocks_no, block_size >> > (deviceKeys, deviceValues, numKeys, table, capacity, success);
    cudaDeviceSynchronize();
    int *hostValues = 0;
    hostValues = (int *) malloc(numKeys * sizeof(int));
    cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceKeys);
    cudaFree(deviceValues);
    return hostValues;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
    return currentSize / (1.0 * capacity); // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
