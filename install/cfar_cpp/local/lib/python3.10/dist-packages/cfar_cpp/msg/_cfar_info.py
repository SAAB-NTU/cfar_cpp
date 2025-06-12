# generated from rosidl_generator_py/resource/_idl.py.em
# with input from cfar_cpp:msg/CfarInfo.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_CfarInfo(type):
    """Metaclass of message 'CfarInfo'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('cfar_cpp')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cfar_cpp.msg.CfarInfo')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__cfar_info
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__cfar_info
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__cfar_info
            cls._TYPE_SUPPORT = module.type_support_msg__msg__cfar_info
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__cfar_info

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class CfarInfo(metaclass=Metaclass_CfarInfo):
    """Message class 'CfarInfo'."""

    __slots__ = [
        '_mode',
        '_train_cells',
        '_guard_cells',
        '_false_alarm_rate',
        '_threshold_factor',
    ]

    _fields_and_field_types = {
        'mode': 'string',
        'train_cells': 'int64',
        'guard_cells': 'int64',
        'false_alarm_rate': 'double',
        'threshold_factor': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.mode = kwargs.get('mode', str())
        self.train_cells = kwargs.get('train_cells', int())
        self.guard_cells = kwargs.get('guard_cells', int())
        self.false_alarm_rate = kwargs.get('false_alarm_rate', float())
        self.threshold_factor = kwargs.get('threshold_factor', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.mode != other.mode:
            return False
        if self.train_cells != other.train_cells:
            return False
        if self.guard_cells != other.guard_cells:
            return False
        if self.false_alarm_rate != other.false_alarm_rate:
            return False
        if self.threshold_factor != other.threshold_factor:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def mode(self):
        """Message field 'mode'."""
        return self._mode

    @mode.setter
    def mode(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'mode' field must be of type 'str'"
        self._mode = value

    @builtins.property
    def train_cells(self):
        """Message field 'train_cells'."""
        return self._train_cells

    @train_cells.setter
    def train_cells(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'train_cells' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'train_cells' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._train_cells = value

    @builtins.property
    def guard_cells(self):
        """Message field 'guard_cells'."""
        return self._guard_cells

    @guard_cells.setter
    def guard_cells(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'guard_cells' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'guard_cells' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._guard_cells = value

    @builtins.property
    def false_alarm_rate(self):
        """Message field 'false_alarm_rate'."""
        return self._false_alarm_rate

    @false_alarm_rate.setter
    def false_alarm_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'false_alarm_rate' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'false_alarm_rate' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._false_alarm_rate = value

    @builtins.property
    def threshold_factor(self):
        """Message field 'threshold_factor'."""
        return self._threshold_factor

    @threshold_factor.setter
    def threshold_factor(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'threshold_factor' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'threshold_factor' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._threshold_factor = value
