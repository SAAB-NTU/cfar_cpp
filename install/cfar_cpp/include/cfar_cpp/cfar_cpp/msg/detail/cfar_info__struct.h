// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice

#ifndef CFAR_CPP__MSG__DETAIL__CFAR_INFO__STRUCT_H_
#define CFAR_CPP__MSG__DETAIL__CFAR_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'mode'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/CfarInfo in the package cfar_cpp.
typedef struct cfar_cpp__msg__CfarInfo
{
  rosidl_runtime_c__String mode;
  int64_t train_cells;
  int64_t guard_cells;
  double false_alarm_rate;
  double threshold_factor;
} cfar_cpp__msg__CfarInfo;

// Struct for a sequence of cfar_cpp__msg__CfarInfo.
typedef struct cfar_cpp__msg__CfarInfo__Sequence
{
  cfar_cpp__msg__CfarInfo * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cfar_cpp__msg__CfarInfo__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CFAR_CPP__MSG__DETAIL__CFAR_INFO__STRUCT_H_
