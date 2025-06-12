// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice

#ifndef CFAR_CPP__MSG__DETAIL__CFAR_INFO__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define CFAR_CPP__MSG__DETAIL__CFAR_INFO__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "cfar_cpp/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "cfar_cpp/msg/detail/cfar_info__struct.hpp"

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

#include "fastcdr/Cdr.h"

namespace cfar_cpp
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cfar_cpp
cdr_serialize(
  const cfar_cpp::msg::CfarInfo & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cfar_cpp
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  cfar_cpp::msg::CfarInfo & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cfar_cpp
get_serialized_size(
  const cfar_cpp::msg::CfarInfo & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cfar_cpp
max_serialized_size_CfarInfo(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace cfar_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cfar_cpp
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, cfar_cpp, msg, CfarInfo)();

#ifdef __cplusplus
}
#endif

#endif  // CFAR_CPP__MSG__DETAIL__CFAR_INFO__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
