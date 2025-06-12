// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice

#ifndef CFAR_CPP__MSG__DETAIL__CFAR_INFO__BUILDER_HPP_
#define CFAR_CPP__MSG__DETAIL__CFAR_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "cfar_cpp/msg/detail/cfar_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace cfar_cpp
{

namespace msg
{

namespace builder
{

class Init_CfarInfo_threshold_factor
{
public:
  explicit Init_CfarInfo_threshold_factor(::cfar_cpp::msg::CfarInfo & msg)
  : msg_(msg)
  {}
  ::cfar_cpp::msg::CfarInfo threshold_factor(::cfar_cpp::msg::CfarInfo::_threshold_factor_type arg)
  {
    msg_.threshold_factor = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cfar_cpp::msg::CfarInfo msg_;
};

class Init_CfarInfo_false_alarm_rate
{
public:
  explicit Init_CfarInfo_false_alarm_rate(::cfar_cpp::msg::CfarInfo & msg)
  : msg_(msg)
  {}
  Init_CfarInfo_threshold_factor false_alarm_rate(::cfar_cpp::msg::CfarInfo::_false_alarm_rate_type arg)
  {
    msg_.false_alarm_rate = std::move(arg);
    return Init_CfarInfo_threshold_factor(msg_);
  }

private:
  ::cfar_cpp::msg::CfarInfo msg_;
};

class Init_CfarInfo_guard_cells
{
public:
  explicit Init_CfarInfo_guard_cells(::cfar_cpp::msg::CfarInfo & msg)
  : msg_(msg)
  {}
  Init_CfarInfo_false_alarm_rate guard_cells(::cfar_cpp::msg::CfarInfo::_guard_cells_type arg)
  {
    msg_.guard_cells = std::move(arg);
    return Init_CfarInfo_false_alarm_rate(msg_);
  }

private:
  ::cfar_cpp::msg::CfarInfo msg_;
};

class Init_CfarInfo_train_cells
{
public:
  explicit Init_CfarInfo_train_cells(::cfar_cpp::msg::CfarInfo & msg)
  : msg_(msg)
  {}
  Init_CfarInfo_guard_cells train_cells(::cfar_cpp::msg::CfarInfo::_train_cells_type arg)
  {
    msg_.train_cells = std::move(arg);
    return Init_CfarInfo_guard_cells(msg_);
  }

private:
  ::cfar_cpp::msg::CfarInfo msg_;
};

class Init_CfarInfo_mode
{
public:
  Init_CfarInfo_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CfarInfo_train_cells mode(::cfar_cpp::msg::CfarInfo::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_CfarInfo_train_cells(msg_);
  }

private:
  ::cfar_cpp::msg::CfarInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cfar_cpp::msg::CfarInfo>()
{
  return cfar_cpp::msg::builder::Init_CfarInfo_mode();
}

}  // namespace cfar_cpp

#endif  // CFAR_CPP__MSG__DETAIL__CFAR_INFO__BUILDER_HPP_
