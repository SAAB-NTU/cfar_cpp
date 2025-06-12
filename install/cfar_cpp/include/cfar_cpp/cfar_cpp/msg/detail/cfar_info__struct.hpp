// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice

#ifndef CFAR_CPP__MSG__DETAIL__CFAR_INFO__STRUCT_HPP_
#define CFAR_CPP__MSG__DETAIL__CFAR_INFO__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__cfar_cpp__msg__CfarInfo __attribute__((deprecated))
#else
# define DEPRECATED__cfar_cpp__msg__CfarInfo __declspec(deprecated)
#endif

namespace cfar_cpp
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CfarInfo_
{
  using Type = CfarInfo_<ContainerAllocator>;

  explicit CfarInfo_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = "";
      this->train_cells = 0ll;
      this->guard_cells = 0ll;
      this->false_alarm_rate = 0.0;
      this->threshold_factor = 0.0;
    }
  }

  explicit CfarInfo_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mode(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = "";
      this->train_cells = 0ll;
      this->guard_cells = 0ll;
      this->false_alarm_rate = 0.0;
      this->threshold_factor = 0.0;
    }
  }

  // field types and members
  using _mode_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _mode_type mode;
  using _train_cells_type =
    int64_t;
  _train_cells_type train_cells;
  using _guard_cells_type =
    int64_t;
  _guard_cells_type guard_cells;
  using _false_alarm_rate_type =
    double;
  _false_alarm_rate_type false_alarm_rate;
  using _threshold_factor_type =
    double;
  _threshold_factor_type threshold_factor;

  // setters for named parameter idiom
  Type & set__mode(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->mode = _arg;
    return *this;
  }
  Type & set__train_cells(
    const int64_t & _arg)
  {
    this->train_cells = _arg;
    return *this;
  }
  Type & set__guard_cells(
    const int64_t & _arg)
  {
    this->guard_cells = _arg;
    return *this;
  }
  Type & set__false_alarm_rate(
    const double & _arg)
  {
    this->false_alarm_rate = _arg;
    return *this;
  }
  Type & set__threshold_factor(
    const double & _arg)
  {
    this->threshold_factor = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cfar_cpp::msg::CfarInfo_<ContainerAllocator> *;
  using ConstRawPtr =
    const cfar_cpp::msg::CfarInfo_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cfar_cpp::msg::CfarInfo_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cfar_cpp::msg::CfarInfo_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cfar_cpp::msg::CfarInfo_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cfar_cpp::msg::CfarInfo_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cfar_cpp::msg::CfarInfo_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cfar_cpp::msg::CfarInfo_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cfar_cpp::msg::CfarInfo_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cfar_cpp::msg::CfarInfo_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cfar_cpp__msg__CfarInfo
    std::shared_ptr<cfar_cpp::msg::CfarInfo_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cfar_cpp__msg__CfarInfo
    std::shared_ptr<cfar_cpp::msg::CfarInfo_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CfarInfo_ & other) const
  {
    if (this->mode != other.mode) {
      return false;
    }
    if (this->train_cells != other.train_cells) {
      return false;
    }
    if (this->guard_cells != other.guard_cells) {
      return false;
    }
    if (this->false_alarm_rate != other.false_alarm_rate) {
      return false;
    }
    if (this->threshold_factor != other.threshold_factor) {
      return false;
    }
    return true;
  }
  bool operator!=(const CfarInfo_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CfarInfo_

// alias to use template instance with default allocator
using CfarInfo =
  cfar_cpp::msg::CfarInfo_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace cfar_cpp

#endif  // CFAR_CPP__MSG__DETAIL__CFAR_INFO__STRUCT_HPP_
