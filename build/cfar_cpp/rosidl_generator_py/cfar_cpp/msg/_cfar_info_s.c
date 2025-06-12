// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from cfar_cpp:msg/CfarInfo.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "cfar_cpp/msg/detail/cfar_info__struct.h"
#include "cfar_cpp/msg/detail/cfar_info__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool cfar_cpp__msg__cfar_info__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[33];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("cfar_cpp.msg._cfar_info.CfarInfo", full_classname_dest, 32) == 0);
  }
  cfar_cpp__msg__CfarInfo * ros_message = _ros_message;
  {  // mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "mode");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->mode, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // train_cells
    PyObject * field = PyObject_GetAttrString(_pymsg, "train_cells");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->train_cells = PyLong_AsLongLong(field);
    Py_DECREF(field);
  }
  {  // guard_cells
    PyObject * field = PyObject_GetAttrString(_pymsg, "guard_cells");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->guard_cells = PyLong_AsLongLong(field);
    Py_DECREF(field);
  }
  {  // false_alarm_rate
    PyObject * field = PyObject_GetAttrString(_pymsg, "false_alarm_rate");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->false_alarm_rate = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // threshold_factor
    PyObject * field = PyObject_GetAttrString(_pymsg, "threshold_factor");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->threshold_factor = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * cfar_cpp__msg__cfar_info__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of CfarInfo */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("cfar_cpp.msg._cfar_info");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "CfarInfo");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  cfar_cpp__msg__CfarInfo * ros_message = (cfar_cpp__msg__CfarInfo *)raw_ros_message;
  {  // mode
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->mode.data,
      strlen(ros_message->mode.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "mode", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // train_cells
    PyObject * field = NULL;
    field = PyLong_FromLongLong(ros_message->train_cells);
    {
      int rc = PyObject_SetAttrString(_pymessage, "train_cells", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // guard_cells
    PyObject * field = NULL;
    field = PyLong_FromLongLong(ros_message->guard_cells);
    {
      int rc = PyObject_SetAttrString(_pymessage, "guard_cells", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // false_alarm_rate
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->false_alarm_rate);
    {
      int rc = PyObject_SetAttrString(_pymessage, "false_alarm_rate", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // threshold_factor
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->threshold_factor);
    {
      int rc = PyObject_SetAttrString(_pymessage, "threshold_factor", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
