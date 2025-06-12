// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice

#ifndef CFAR_CPP__MSG__DETAIL__CFAR_INFO__FUNCTIONS_H_
#define CFAR_CPP__MSG__DETAIL__CFAR_INFO__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "cfar_cpp/msg/rosidl_generator_c__visibility_control.h"

#include "cfar_cpp/msg/detail/cfar_info__struct.h"

/// Initialize msg/CfarInfo message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * cfar_cpp__msg__CfarInfo
 * )) before or use
 * cfar_cpp__msg__CfarInfo__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
bool
cfar_cpp__msg__CfarInfo__init(cfar_cpp__msg__CfarInfo * msg);

/// Finalize msg/CfarInfo message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
void
cfar_cpp__msg__CfarInfo__fini(cfar_cpp__msg__CfarInfo * msg);

/// Create msg/CfarInfo message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * cfar_cpp__msg__CfarInfo__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
cfar_cpp__msg__CfarInfo *
cfar_cpp__msg__CfarInfo__create();

/// Destroy msg/CfarInfo message.
/**
 * It calls
 * cfar_cpp__msg__CfarInfo__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
void
cfar_cpp__msg__CfarInfo__destroy(cfar_cpp__msg__CfarInfo * msg);

/// Check for msg/CfarInfo message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
bool
cfar_cpp__msg__CfarInfo__are_equal(const cfar_cpp__msg__CfarInfo * lhs, const cfar_cpp__msg__CfarInfo * rhs);

/// Copy a msg/CfarInfo message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
bool
cfar_cpp__msg__CfarInfo__copy(
  const cfar_cpp__msg__CfarInfo * input,
  cfar_cpp__msg__CfarInfo * output);

/// Initialize array of msg/CfarInfo messages.
/**
 * It allocates the memory for the number of elements and calls
 * cfar_cpp__msg__CfarInfo__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
bool
cfar_cpp__msg__CfarInfo__Sequence__init(cfar_cpp__msg__CfarInfo__Sequence * array, size_t size);

/// Finalize array of msg/CfarInfo messages.
/**
 * It calls
 * cfar_cpp__msg__CfarInfo__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
void
cfar_cpp__msg__CfarInfo__Sequence__fini(cfar_cpp__msg__CfarInfo__Sequence * array);

/// Create array of msg/CfarInfo messages.
/**
 * It allocates the memory for the array and calls
 * cfar_cpp__msg__CfarInfo__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
cfar_cpp__msg__CfarInfo__Sequence *
cfar_cpp__msg__CfarInfo__Sequence__create(size_t size);

/// Destroy array of msg/CfarInfo messages.
/**
 * It calls
 * cfar_cpp__msg__CfarInfo__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
void
cfar_cpp__msg__CfarInfo__Sequence__destroy(cfar_cpp__msg__CfarInfo__Sequence * array);

/// Check for msg/CfarInfo message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
bool
cfar_cpp__msg__CfarInfo__Sequence__are_equal(const cfar_cpp__msg__CfarInfo__Sequence * lhs, const cfar_cpp__msg__CfarInfo__Sequence * rhs);

/// Copy an array of msg/CfarInfo messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_cfar_cpp
bool
cfar_cpp__msg__CfarInfo__Sequence__copy(
  const cfar_cpp__msg__CfarInfo__Sequence * input,
  cfar_cpp__msg__CfarInfo__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CFAR_CPP__MSG__DETAIL__CFAR_INFO__FUNCTIONS_H_
