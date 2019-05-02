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
   // printf("Offset = %d\n", offset);
    unsigned long long a = primeList[MAX_HASH_PARAM-offset];
    unsigned long long b = primeList[MAX_HASH_PARAM-offset-1];
    return (a*key + b) % INT_MAX;
}

__global__ void cuckooInsert( int* keys, int* values, int numKeys, Bucket *table, int capacity)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numKeys) {
        int key = keys[tid];
        int value = values[tid];
        //printf("Key val insert : %d - %d\n", key, value);
        Bucket newValue = (((static_cast<Bucket>(key) << 32)) | (static_cast<Bucket>(value)));
       // printf("Key = %d  ", (newValue & 0xffffffff00000000)>>32);
        //printf("Val = %d\n", (newValue & 0x00000000ffffffff)); 
        
        int idx[5];
        idx[0] = hashFunc(key, 0) % capacity;
       // printf("Capacity = %d idx[0] = %d\n", capacity, idx[0]);
        for (int i = 0; i < 1000; ++i) {
           // printf("In for\n\n");           
            newValue = atomicExch(&table[idx[0]], newValue);
           
	//    printf("After atomic Exch\n\n");
           // printf("table[%d].val = %d\n", idx[0], table[idx[0]] & 0x00000000ffffffff);
          //  auto aux1 = ((table[idx[0]] & 0xffffffff00000000)>>32);
           // auto aux2 = table[idx[0]] & 0x00000000ffffffff;
	    //printf("table[%d].key = %d\n", idx[0], aux1);
           // printf("table[%d].val = %d\n", idx[0], aux2);
	    if ((newValue & 0xffffffff00000000) >> 32 == KEY_INVALID) {
		   return;
            }

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

        printf("tid %d: Insert (%u,%u) failed\n", tid, key, value);
    }
    return;
}

__global__ void cuckooGet( int* keys, int* values, int numKeys, Bucket *table, int capacity)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numKeys) {
        int key = keys[tid];

        // Compute all possible locations
        uint32_t idx[4];
        idx[0] = hashFunc(key, 0 );
        idx[1] = hashFunc(key, 1 );
        idx[2] = hashFunc(key, 2 );
        idx[3] = hashFunc(key, 3 );

        Bucket entry;
        for( int i = 0; i < 4; ++i )
        {
            entry = static_cast<Bucket> (table[idx[i] % capacity]);
            Key k = (entry & 0xffffffff00000000) >> (8 * sizeof(Value));
            if( k == key ) {
                int val = (entry & 0x00000000ffffffff);
		values[tid] = val;
                return;
            }
            if( k == KEY_INVALID )
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
    currentSize = 0;
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
    capacity = numBucketsReshape;
    table = newTable;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {

    currentSize += numKeys;
    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
   printf("Insertttt\n");
    for (int i = 0; i < numKeys; i++) {
        printf("<key, value> : <%d, %d>\n", keys[i], values[i]);
    }
    const int block_size = 64;
    int blocks_no = numKeys / block_size;

    if (numKeys % block_size)
        ++blocks_no;

    // Launch the kernel
    cuckooInsert <<< blocks_no, block_size >>> (deviceKeys, deviceValues, numKeys, table, capacity);
cudaDeviceSynchronize();

    Bucket *hostValues = 0;

    hostValues = (Bucket*) malloc(capacity * sizeof(Bucket));
    cudaMemcpy(hostValues, table, capacity * sizeof(Bucket), cudaMemcpyDeviceToHost);
    printf("After insert in table:\n");
    for (int i = 0; i < capacity; i++) {
        printf("Key = %d -> ", (hostValues[i] & 0xffffffff00000000) >> 32);
        printf("Value = %d\n", (hostValues[i] & 0x00000000ffffffff));
    }

    cudaFree(deviceKeys);
    cudaFree(deviceValues);
    deviceKeys = 0;
    deviceValues = 0;
    cudaDeviceSynchronize();
    return false;
}

/* GET BATCH
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
    Bucket *hostValues2 = 0;

    hostValues2 = (Bucket*) malloc(capacity * sizeof(Bucket));
    cudaMemcpy(hostValues2, table, capacity, cudaMemcpyDeviceToHost);
   // printf("Gettttt\n");
   // for (int i = 0; i < capacity; i++) {
     //   printf("<key, value> : <%d, %d>\n", hostValues2[i] & 0xffff0000, hostValues2[i] & 0x0000ffff);
   // }


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
//    cudaDeviceSynchronize();
   // cudaFree(deviceKeys);
   // cudaFree(deviceValues);
  // printf("REturnnnnn\n");
  // for (int i = 0; i < numKeys; i++) {
    //    printf("<key, value> : <%d, %d>\n", keys[i], hostValues[i]);
  // }
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
