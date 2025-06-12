// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice
#include "cfar_cpp/msg/detail/cfar_info__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `mode`
#include "rosidl_runtime_c/string_functions.h"

bool
cfar_cpp__msg__CfarInfo__init(cfar_cpp__msg__CfarInfo * msg)
{
  if (!msg) {
    return false;
  }
  // mode
  if (!rosidl_runtime_c__String__init(&msg->mode)) {
    cfar_cpp__msg__CfarInfo__fini(msg);
    return false;
  }
  // train_cells
  // guard_cells
  // false_alarm_rate
  // threshold_factor
  return true;
}

void
cfar_cpp__msg__CfarInfo__fini(cfar_cpp__msg__CfarInfo * msg)
{
  if (!msg) {
    return;
  }
  // mode
  rosidl_runtime_c__String__fini(&msg->mode);
  // train_cells
  // guard_cells
  // false_alarm_rate
  // threshold_factor
}

bool
cfar_cpp__msg__CfarInfo__are_equal(const cfar_cpp__msg__CfarInfo * lhs, const cfar_cpp__msg__CfarInfo * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // mode
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->mode), &(rhs->mode)))
  {
    return false;
  }
  // train_cells
  if (lhs->train_cells != rhs->train_cells) {
    return false;
  }
  // guard_cells
  if (lhs->guard_cells != rhs->guard_cells) {
    return false;
  }
  // false_alarm_rate
  if (lhs->false_alarm_rate != rhs->false_alarm_rate) {
    return false;
  }
  // threshold_factor
  if (lhs->threshold_factor != rhs->threshold_factor) {
    return false;
  }
  return true;
}

bool
cfar_cpp__msg__CfarInfo__copy(
  const cfar_cpp__msg__CfarInfo * input,
  cfar_cpp__msg__CfarInfo * output)
{
  if (!input || !output) {
    return false;
  }
  // mode
  if (!rosidl_runtime_c__String__copy(
      &(input->mode), &(output->mode)))
  {
    return false;
  }
  // train_cells
  output->train_cells = input->train_cells;
  // guard_cells
  output->guard_cells = input->guard_cells;
  // false_alarm_rate
  output->false_alarm_rate = input->false_alarm_rate;
  // threshold_factor
  output->threshold_factor = input->threshold_factor;
  return true;
}

cfar_cpp__msg__CfarInfo *
cfar_cpp__msg__CfarInfo__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cfar_cpp__msg__CfarInfo * msg = (cfar_cpp__msg__CfarInfo *)allocator.allocate(sizeof(cfar_cpp__msg__CfarInfo), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cfar_cpp__msg__CfarInfo));
  bool success = cfar_cpp__msg__CfarInfo__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cfar_cpp__msg__CfarInfo__destroy(cfar_cpp__msg__CfarInfo * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cfar_cpp__msg__CfarInfo__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cfar_cpp__msg__CfarInfo__Sequence__init(cfar_cpp__msg__CfarInfo__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cfar_cpp__msg__CfarInfo * data = NULL;

  if (size) {
    data = (cfar_cpp__msg__CfarInfo *)allocator.zero_allocate(size, sizeof(cfar_cpp__msg__CfarInfo), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cfar_cpp__msg__CfarInfo__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cfar_cpp__msg__CfarInfo__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
cfar_cpp__msg__CfarInfo__Sequence__fini(cfar_cpp__msg__CfarInfo__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      cfar_cpp__msg__CfarInfo__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

cfar_cpp__msg__CfarInfo__Sequence *
cfar_cpp__msg__CfarInfo__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cfar_cpp__msg__CfarInfo__Sequence * array = (cfar_cpp__msg__CfarInfo__Sequence *)allocator.allocate(sizeof(cfar_cpp__msg__CfarInfo__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cfar_cpp__msg__CfarInfo__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cfar_cpp__msg__CfarInfo__Sequence__destroy(cfar_cpp__msg__CfarInfo__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cfar_cpp__msg__CfarInfo__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cfar_cpp__msg__CfarInfo__Sequence__are_equal(const cfar_cpp__msg__CfarInfo__Sequence * lhs, const cfar_cpp__msg__CfarInfo__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cfar_cpp__msg__CfarInfo__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cfar_cpp__msg__CfarInfo__Sequence__copy(
  const cfar_cpp__msg__CfarInfo__Sequence * input,
  cfar_cpp__msg__CfarInfo__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cfar_cpp__msg__CfarInfo);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    cfar_cpp__msg__CfarInfo * data =
      (cfar_cpp__msg__CfarInfo *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cfar_cpp__msg__CfarInfo__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          cfar_cpp__msg__CfarInfo__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!cfar_cpp__msg__CfarInfo__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
