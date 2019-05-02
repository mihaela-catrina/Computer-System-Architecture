#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__device__ int hashFunc(int key, int capacity, int offset = 0) {
    if (offset == 0) return hash1(key, capacity);
    if (offset == 1) return hash2(key, capacity);
    if (offset == 2) return hash3(key, capacity);
    return hash3(key, capacity);
}

/**/
__global__ void cuckooInsert(int *keys, int *values, int numKeys, Bucket *table, int capacity, int *currentSize, int *updates) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numKeys) {
        int key = keys[tid];
        int value = values[tid];
        // create a new 64-bit value (key | value)
        Bucket newValue = (((static_cast<Bucket>(key) << INT_BITS)) | (static_cast<Bucket>(value)));
        
        // Compute all possible locations
	int idx[4];
	idx[0] = hashFunc(key, capacity, 0);
        idx[1] = hashFunc(key, capacity, 1);
        idx[2] = hashFunc(key, capacity, 2);
        
        Bucket entry;
        // verify if the key is already in the hashmap
        for (int i = 0; i < 3; ++i) {
            entry = static_cast<Bucket> (table[idx[i]]);
            Key k = (entry & HIGH) >> INT_BITS;
            // Update
            if (k == key) {
                newValue = atomicExch(&table[idx[i]], newValue);
                atomicAdd(updates, 1);
                return;
            }
        }
        // Othrewise try to insert the key in the table
        idx[0] = hashFunc(key, capacity, 0);
        // Max probe heuristic
        for (int i = 0; i < 7 *log2f(numKeys); ++i) {
	    // Exchange
            newValue = atomicExch(&table[idx[0]], newValue);
            if ((newValue & HIGH) >> INT_BITS == KEY_INVALID) {
              	 atomicAdd(currentSize, 1);
		 return;
            }
           
            key = (newValue & HIGH) >> INT_BITS;
            // Otherwise find a new location for the displaced item
            int last_loc = idx[0];
            idx[0] = hashFunc(key, capacity, 0);
            idx[1] = hashFunc(key, capacity, 1);
            idx[2] = hashFunc(key, capacity, 2);
            
            for (int i = 1; i >= 0; --i)
                idx[0] = (last_loc == idx[i] ? idx[i + 1] : idx[0]);

        }
    }
    return;
}

/**/
__global__ void cuckooGet(int *keys, int *values, int numKeys, Bucket *table, int capacity) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numKeys) {
        int key = keys[tid];

        // Compute all possible locations
        int loc[4];
        loc[0] = hashFunc(key, capacity, 0);
        loc[1] = hashFunc(key, capacity, 1);
        loc[2] = hashFunc(key, capacity, 2);

        Bucket entry;
        for (int i = 0; i < 3; ++i) {
            entry = table[loc[i]];
            Key k = (entry & HIGH) >> INT_BITS;
            // Get value
	    if (k == key) {
                int val = (entry & LOW);
                values[tid] = val;
                return;
            }
        } 
    }

    return;
}


/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {

    capacity = 0;
    cudaMallocManaged(&currentSize, 4);
    cudaMallocManaged(&updates, 4);
    *updates = 0;
    *currentSize = 0;
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
    oldKeys = (int*)malloc(*currentSize * sizeof(int));
    oldValues = (int*)malloc(*currentSize * sizeof(int));
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // Store the entries that are already present in the table (used in case of rehash)
    Bucket *tableValues = 0;
    tableValues = (Bucket *) malloc(capacity * sizeof(Bucket));
    cudaMemcpy(tableValues, table, capacity * sizeof(Bucket), cudaMemcpyDeviceToHost);
    int index = 0;
    for (int i = 0; i < capacity; i++) {
        if(tableValues[i] != KEY_INVALID) {
            oldValues[index] = (tableValues[i] & LOW);
            oldKeys[index] = (tableValues[i] & HIGH) >> INT_BITS;
            index++;
	}
        
    }
    oldSize = *currentSize;
    *updates = 0;
    free(tableValues);

    // Calculate reasonable block dimensions
    const int block_size = 64;
    int blocks_no = numKeys / block_size;
    if (numKeys % block_size)
        ++blocks_no;
    
    // Launch the kernel
    cuckooInsert <<< blocks_no, block_size >>> (deviceKeys, deviceValues, numKeys, table, capacity, currentSize, updates);
    cudaDeviceSynchronize();
     // If all the values have not been successfully inserted
     if (*currentSize != oldSize + numKeys - *updates) {
        rehash(keys, values, numKeys);
    } else {
        // Copy values to the host part
        Bucket *hostValues = 0;
        hostValues = (Bucket *) malloc(capacity * sizeof(Bucket));
        cudaMemcpy(hostValues, table, capacity * sizeof(Bucket), cudaMemcpyDeviceToHost);
        // Free
        cudaFree(deviceKeys);
        cudaFree(deviceValues);
        free(oldKeys);
        free(oldValues);
        deviceKeys = 0;
        deviceValues = 0;
        oldKeys = 0;
        oldValues = 0;
    }

    return false;
}

void GpuHashTable::rehash(int *keys, int*values, int numKeys) {
    Bucket *newTable = NULL;
    Bucket *aux = 0;
    cudaMalloc((void **) &newTable, 2 * capacity * sizeof(Bucket));
    cudaMemset(newTable, KEY_INVALID, 2 * capacity * sizeof(Bucket));
    capacity *= 2;
    *currentSize = 0;

    aux = table;
    table = newTable;
    cudaFree(aux);

    if (oldSize != 0) {
   	 insertBatch(oldKeys, oldValues, oldSize);
    }
    insertBatch(keys, values, numKeys);

}

/* GET BATCH
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    const int block_size = 64;
    int blocks_no = numKeys / block_size;

    if (numKeys % block_size)
        ++blocks_no;
    cuckooGet <<< blocks_no, block_size >>> (deviceKeys, deviceValues, numKeys, table, capacity);
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
    return *currentSize / (1.0 * capacity); // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
