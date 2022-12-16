// Implementing hash table in C
// Idea taken from https://www.programiz.com/dsa/hash-table
#include "HashFunc.h"

int checkPrime(int n)
{
  int i;
  if (n == 1 || n == 0)
  {
  return 0;
  }
  for (i = 2; i < n / 2; i++)
  {
  if (n % i == 0)
  {
    return 0;
  }
  }
  return 1;
}
int getPrime(int n)
{
  if (n % 2 == 0)
  {
  n++;
  }
  while (!checkPrime(n))
  {
  n += 2;
  }
  return n;
}

void init_array(quant* array, int capacity)
{
  capacity = getPrime(capacity);
  array = (struct set *)malloc(capacity * sizeof(struct set));
  for (int i = 0; i < capacity; i++)
  {
  array[i].key = 0;
  array[i].data = 0;
  }
}

void insert(quant* array, fp32 key, ui8 data)
{
  int index = static_cast< int >key;
  if (array[index].data == 0)
  {
  array[index].key = key;
  array[index].data = data;

  printf("\n Key (%d) has been inserted \n", key);
  }
  else if (array[index].key == key)
  {
  array[index].data = data;
  }
  else
  {
  printf("\n Collision occured  \n");
  }
}

void remove_element(quant* array, fp32 key, int capacity)
{
  int index = static_cast< int >key;
  if (array[index].data == 0)
  {
  printf("\n This key does not exist \n");
  }
  else
  {
  array[index].key = 0;
  array[index].data = 0;
  printf("\n Key (%d) has been removed \n", key);
  }
}

ui8 read_hash(quant* array, fp32 key, int capacity)
{
  int index = static_cast< int >key;
  if (array[index].data == 0)
  {
  printf("\n This key does not exist \n");
  }
  else
  {
  return array[index].data;
  }
}

void display(quant* array, int capacity)
{
  int i;
  for (i = 0; i < capacity; i++)
  {
  if (array[i].data == 0)
  {
    printf("\n array[%d]: / ", i);
  }
  else
  {
    printf("\n key: %d array[%d]: %d \t", array[i].key, i, array[i].data);
  }
  }
}

void d_init_array(dequant* array, int capacity)
{
  capacity = getPrime(capacity);
  array = (struct set *)malloc(capacity * sizeof(struct set));
  for (int i = 0; i < capacity; i++)
  {
  array[i].key = 0;
  array[i].data = 0;
  }
}

void d_insert(dequant* array, ui8 key, fp32 data)
{
  int index = static_cast< int >key;
  if (array[index].data == 0)
  {
  array[index].key = key;
  array[index].data = data;

  printf("\n Key (%d) has been inserted \n", key);
  }
  else if (array[index].key == key)
  {
  array[index].data = data;
  }
  else
  {
  printf("\n Collision occured  \n");
  }
}

void d_remove_element(dequant* array, ui8 key)
{
  int index = static_cast< int >key;
  if (array[index].data == 0)
  {
  printf("\n This key does not exist \n");
  }
  else
  {
  array[index].key = 0;
  array[index].data = 0;
  printf("\n Key (%d) has been removed \n", key);
  }
}

f32 d_read_hash(dequant* array, ui8 key)
{
  int index = static_cast< int >key;
  if (array[index].data == 0)
  {
  printf("\n This key does not exist \n");
  }
  else
  {
  return array[index].data;
  }
}

void d_display(dequant* array, int capacity)
{
  int i;
  for (i = 0; i < capacity; i++)
  {
  if (array[i].data == 0)
  {
    printf("\n array[%d]: / ", i);
  }
  else
  {
    printf("\n key: %d array[%d]: %d \t", array[i].key, i, array[i].data);
  }
  }
}