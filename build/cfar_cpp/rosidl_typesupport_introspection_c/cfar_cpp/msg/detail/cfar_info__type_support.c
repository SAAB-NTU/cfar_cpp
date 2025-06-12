// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cfar_cpp/msg/detail/cfar_info__rosidl_typesupport_introspection_c.h"
#include "cfar_cpp/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cfar_cpp/msg/detail/cfar_info__functions.h"
#include "cfar_cpp/msg/detail/cfar_info__struct.h"


// Include directives for member types
// Member `mode`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cfar_cpp__msg__CfarInfo__init(message_memory);
}

void cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_fini_function(void * message_memory)
{
  cfar_cpp__msg__CfarInfo__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_message_member_array[5] = {
  {
    "mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cfar_cpp__msg__CfarInfo, mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "train_cells",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cfar_cpp__msg__CfarInfo, train_cells),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "guard_cells",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cfar_cpp__msg__CfarInfo, guard_cells),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "false_alarm_rate",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cfar_cpp__msg__CfarInfo, false_alarm_rate),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "threshold_factor",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cfar_cpp__msg__CfarInfo, threshold_factor),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_message_members = {
  "cfar_cpp__msg",  // message namespace
  "CfarInfo",  // message name
  5,  // number of fields
  sizeof(cfar_cpp__msg__CfarInfo),
  cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_message_member_array,  // message members
  cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_init_function,  // function to initialize message memory (memory has to be allocated)
  cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_message_type_support_handle = {
  0,
  &cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cfar_cpp
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cfar_cpp, msg, CfarInfo)() {
  if (!cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_message_type_support_handle.typesupport_identifier) {
    cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &cfar_cpp__msg__CfarInfo__rosidl_typesupport_introspection_c__CfarInfo_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
