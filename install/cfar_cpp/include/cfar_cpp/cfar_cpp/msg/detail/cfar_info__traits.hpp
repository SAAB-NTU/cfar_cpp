// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice

#ifndef CFAR_CPP__MSG__DETAIL__CFAR_INFO__TRAITS_HPP_
#define CFAR_CPP__MSG__DETAIL__CFAR_INFO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "cfar_cpp/msg/detail/cfar_info__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace cfar_cpp
{

namespace msg
{

inline void to_flow_style_yaml(
  const CfarInfo & msg,
  std::ostream & out)
{
  out << "{";
  // member: mode
  {
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << ", ";
  }

  // member: train_cells
  {
    out << "train_cells: ";
    rosidl_generator_traits::value_to_yaml(msg.train_cells, out);
    out << ", ";
  }

  // member: guard_cells
  {
    out << "guard_cells: ";
    rosidl_generator_traits::value_to_yaml(msg.guard_cells, out);
    out << ", ";
  }

  // member: false_alarm_rate
  {
    out << "false_alarm_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.false_alarm_rate, out);
    out << ", ";
  }

  // member: threshold_factor
  {
    out << "threshold_factor: ";
    rosidl_generator_traits::value_to_yaml(msg.threshold_factor, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const CfarInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << "\n";
  }

  // member: train_cells
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "train_cells: ";
    rosidl_generator_traits::value_to_yaml(msg.train_cells, out);
    out << "\n";
  }

  // member: guard_cells
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "guard_cells: ";
    rosidl_generator_traits::value_to_yaml(msg.guard_cells, out);
    out << "\n";
  }

  // member: false_alarm_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "false_alarm_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.false_alarm_rate, out);
    out << "\n";
  }

  // member: threshold_factor
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "threshold_factor: ";
    rosidl_generator_traits::value_to_yaml(msg.threshold_factor, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CfarInfo & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace cfar_cpp

namespace rosidl_generator_traits
{

[[deprecated("use cfar_cpp::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const cfar_cpp::msg::CfarInfo & msg,
  std::ostream & out, size_t indentation = 0)
{
  cfar_cpp::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use cfar_cpp::msg::to_yaml() instead")]]
inline std::string to_yaml(const cfar_cpp::msg::CfarInfo & msg)
{
  return cfar_cpp::msg::to_yaml(msg);
}

template<>
inline const char * data_type<cfar_cpp::msg::CfarInfo>()
{
  return "cfar_cpp::msg::CfarInfo";
}

template<>
inline const char * name<cfar_cpp::msg::CfarInfo>()
{
  return "cfar_cpp/msg/CfarInfo";
}

template<>
struct has_fixed_size<cfar_cpp::msg::CfarInfo>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cfar_cpp::msg::CfarInfo>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cfar_cpp::msg::CfarInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CFAR_CPP__MSG__DETAIL__CFAR_INFO__TRAITS_HPP_
