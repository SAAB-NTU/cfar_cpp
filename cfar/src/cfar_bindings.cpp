#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cfar.h"
#include "utils/ndarray_converter.h"

namespace py = pybind11;

PYBIND11_MODULE(pycfar, handle) {
     NDArrayConverter::init_numpy();
     handle.doc() = "Python bindings for cfar_cpp - A C++ implementation of Constant False Alarm Rate (CFAR) filter";
     py::class_<CFAR>(handle, "CFAR")
          .def(
              py::init<>(), "Default constructor for CFAR."
          )
          .def(
              py::init<int,int,float>(),
              py::arg("train_cells"), py::arg("guard_cells"), py::arg("Pfa"),
              "Constructor for CFAR with specified train cells, guard cells and Pfa"
          )
          // Accessors
          .def("getTrainCells", &CFAR::getTrainCells, "Gets the number of train cells.")
          .def("getGuardCells", &CFAR::getGuardCells, "Gets the number of guard cells.")
          .def("getPfa", &CFAR::getPfa, "Gets the probability of false alarm (Pfa).")
          .def("getThresholdMultiplier", &CFAR::getThresholdMultiplier, "Gets the threshold multiplier.")
          // Main Functions
          .def("soca_1d_naive", &CFAR::soca_1d_naive,
               py::arg("img"),
               "1D SOCA CFAR with a naive approach.")
          .def("soca_2d_naive", &CFAR::soca_2d_naive,
               py::arg("img"),
               "2D SOCA CFAR with a naive approach.")
          .def("soca_1d", &CFAR::soca_1d,
               py::arg("img"),
               "1D SOCA CFAR")
          .def("soca_2d", &CFAR::soca_2d,
               py::arg("img"),
               "2D SOCA CFAR")
          .def("soca_vert", &CFAR::soca_vert,
               py::arg("img"),
               "Vertical-configuration 2D SOCA CFAR");
}
