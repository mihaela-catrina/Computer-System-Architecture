#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	capacity = size;
    // free slots have <key, value> equal to 0
    CUDA_CALL(cudaMalloc((void**)&table, capacity * sizeof(Bucket)));
    CUDA_CALL(cudaMemset(table, 0, capacity * sizeof(Bucket)));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
    CUDA_CALL(cudaFree(table));
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	return false;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
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
