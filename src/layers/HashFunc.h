#include <stdio.h>
#include <stdlib.h>
#include "../Utils.h"
#include "../Types.h"

struct quant
{
  fp32 key;
  ui8 data;
};

int checkPrime(int n);
int getPrime(int n);
void init_array(quant* array, int capacity);
void insert(quant* array, fp32 key, ui8 data);
ui8 read_hash(quant* array, fp32 key);
void remove_element(quant* array, fp32 key;
void display(quant* array, int capacity);

struct dequant
{
  ui8 key;
  fp32 data;
};

void d_init_array(dequant* array, int capacity);
void d_insert(dequant* array, ui8 key, fp32 data, int capacity);
void d_remove_element(dequant* array, ui8 key);
f32 d_read_hash(dequant* array, ui8 key);
void d_display(dequant* array, int capacity);