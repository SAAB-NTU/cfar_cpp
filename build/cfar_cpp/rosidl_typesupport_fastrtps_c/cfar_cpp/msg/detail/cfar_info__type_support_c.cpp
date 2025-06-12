// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice
#include "cfar_cpp/msg/detail/cfar_info__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "cfar_cpp/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "cfar_cpp/msg/detail/cfar_info__struct.h"
#include "cfar_cpp/msg/detail/cfar_info__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "rosidl_runtime_c/string.h"  // mode
#include "rosidl_runtime_c/string_functions.h"  // mode

// forward declare type support functions


using _CfarInfo__ros_msg_type = cfar_cpp__msg__CfarInfo;

static bool _CfarInfo__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _CfarInfo__ros_msg_type * ros_message = static_cast<const _CfarInfo__ros_msg_type *>(untyped_ros_message);
  // Field name: mode
  {
    const rosidl_runtime_c__String * str = &ros_message->mode;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: train_cells
  {
    cdr << ros_message->train_cells;
  }

  // Field name: guard_cells
  {
    cdr << ros_message->guard_cells;
  }

  // Field name: false_alarm_rate
  {
    cdr << ros_message->false_alarm_rate;
  }

  // Field name: threshold_factor
  {
    cdr << ros_message->threshold_factor;
  }

  return true;
}

static bool _CfarInfo__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _CfarInfo__ros_msg_type * ros_message = static_cast<_CfarInfo__ros_msg_type *>(untyped_ros_message);
  // Field name: mode
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->mode.data) {
      rosidl_runtime_c__String__init(&ros_message->mode);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->mode,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'mode'\n");
      return false;
    }
  }

  // Field name: train_cells
  {
    cdr >> ros_message->train_cells;
  }

  // Field name: guard_cells
  {
    cdr >> ros_message->guard_cells;
  }

  // Field name: false_alarm_rate
  {
    cdr >> ros_message->false_alarm_rate;
  }

  // Field name: threshold_factor
  {
    cdr >> ros_message->threshold_factor;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_cfar_cpp
size_t get_serialized_size_cfar_cpp__msg__CfarInfo(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _CfarInfo__ros_msg_type * ros_message = static_cast<const _CfarInfo__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name mode
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->mode.size + 1);
  // field.name train_cells
  {
    size_t item_size = sizeof(ros_message->train_cells);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name guard_cells
  {
    size_t item_size = sizeof(ros_message->guard_cells);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name false_alarm_rate
  {
    size_t item_size = sizeof(ros_message->false_alarm_rate);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name threshold_factor
  {
    size_t item_size = sizeof(ros_message->threshold_factor);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _CfarInfo__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_cfar_cpp__msg__CfarInfo(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_cfar_cpp
size_t max_serialized_size_cfar_cpp__msg__CfarInfo(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: mode
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: train_cells
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: guard_cells
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: false_alarm_rate
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: threshold_factor
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = cfar_cpp__msg__CfarInfo;
    is_plain =
      (
      offsetof(DataType, threshold_factor) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _CfarInfo__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_cfar_cpp__msg__CfarInfo(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_CfarInfo = {
  "cfar_cpp::msg",
  "CfarInfo",
  _CfarInfo__cdr_serialize,
  _CfarInfo__cdr_deserialize,
  _CfarInfo__get_serialized_size,
  _CfarInfo__max_serialized_size
};

static rosidl_message_type_support_t _CfarInfo__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_CfarInfo,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, cfar_cpp, msg, CfarInfo)() {
  return &_CfarInfo__type_support;
}

#if defined(__cplusplus)
}
#endif
