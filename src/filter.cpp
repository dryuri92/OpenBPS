#include "openbps/filter.h"
#include <sstream>
#include <iostream>
#include <memory>
#include <algorithm>
#include <cstring>
#include "../extern/pugiData/pugixml.h"
#include "openbps/parse.h"
#include "openbps/capi.h"


namespace openbps {

//==============================================================================
// Global variables
//==============================================================================
std::vector<Filter> filters;
std::unique_ptr<MaterialFilter> materialfilter;
std::unique_ptr<TimeFilter> timefilter;

//==============================================================================
// Filters class implementation
//==============================================================================

Filter::Filter(pugi::xml_node node)
{
   type = node.attribute("type").value();
   bins_ = get_node_array<std::string>(node, "filter");
}

//! Apply filter to results
void Filter::apply(const std::vector<std::string>&input,
           std::vector<int>& indices) {
    for (int i = 0; i < input.size(); i++) {
        auto search =
                std::find_if(bins_.begin(), bins_.end(),
                             [&input, i](const std::string& bname) {
            return bname == input[i];
        });
        if (search != bins_.end()) {
            indices.push_back(i);
        }
    }
}

MaterialFilter::MaterialFilter(pugi::xml_node node)
{
   type = "material";
   bins_ = get_node_array<std::string>(node, "filter");
}

//! Apply filter to results
void MaterialFilter::apply(const std::string &matname, bool& isValid) {
    auto search =
            std::find_if(bins_.begin(), bins_.end(),
                         [&matname](const std::string& bname) {
                             return bname == matname;
    });
    isValid = (search != bins_.end());

}

TimeFilter::TimeFilter(pugi::xml_node node)
{
   type = "time";
   bins_ = get_node_array<double>(node, "filter");
   if (bins_.size() % 2 == 1) {
       std::cout << "Intervals number should be even" << std::endl;
       bins_.erase(bins_.begin() + bins_.size() - 1, bins_.end());
   }

}

//! Apply filter to results
void TimeFilter::apply(double dt, int numstep, std::vector<int>& indices) {
    size_t j = 0;
    for (size_t k = 0; k < bins_.size() / 2; k++)
        while(j < numstep) {
            if ((j + 1) * dt <= bins_[2 * k + 1] &&
                    (j + 1) * dt > bins_[2 * k]) {
                 indices.push_back(j);

            }
            if ((j + 1) * dt > bins_[2 * k + 1])
                break;
             j++;
       }
}
//==============================================================================
// Non - class methods implementation
//==============================================================================
//! Reading data from configure.xml
void read_fitlers_from_xml(pugi::xml_node root_node) {
    // Proceed all filters
    for (pugi::xml_node tool : root_node.children("filter")) {
        std::string current_type;
        current_type = tool.attribute("type").value();
        if (current_type == "time") {
            timefilter = std::unique_ptr<TimeFilter>(new TimeFilter(tool));
        } else if (current_type == "material") {
            materialfilter = std::unique_ptr<MaterialFilter>(new MaterialFilter(tool));
        } else {
            Filter f(tool);
            filters.push_back(f);
        }
    }
}

} //namespace openbps

//==============================================================================
// C API
//==============================================================================

//added param input_size for convinience
extern "C" int
openbps_filter_apply (int32_t index, const char** input, size_t input_size, int** indices)
{
  int err = 0;
  if (index >= 0 && index < openbps::filters.size()) {
    std::vector<int> tmp_indices; //clean indices in input (if Im not wrong)
    try {
      openbps::filters[index].apply({input, input + input_size}, tmp_indices);
      *indices = tmp_indices.data();
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  } else {
//    set_errmsg("Index in composition array is out of bounds.");
    return OPENBPS_E_OUT_OF_BOUNDS;
  }
  return err;
}

extern "C" int
openbps_material_filter_apply (const char* mat_name, bool* is_valid)
{
    int err = 0;
    try {
      openbps::materialfilter->apply({mat_name}, *is_valid);
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  return err;
}

extern "C" int
openbps_time_filter_apply (double dt, int numstep, int** indices)
{
    int err = 0;
    std::vector<int> tmp_indices; //clean indices in input (if Im not wrong)
    try {
      openbps::timefilter->apply(dt, numstep, tmp_indices);
      *indices = tmp_indices.data();
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  return err;
}

extern "C" int
openbps_material_filter_get_bins (char*** out)
{
    int err = 0;
    try {
      size_t i = 0;
      for (const auto& el : openbps::materialfilter->bins_) {
        std::strcpy((*out)[i], el.c_str());
        i++;
      }
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  return err;
}

extern "C" int
openbps_material_filter_add_bin(char* bin)
{
    int err = 0;
    try {
      openbps::materialfilter->bins_.push_back({bin});
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  return err;
}

extern "C" int
openbps_material_filter_delete_bin(char* bin)
{
    int err = 0;
    try {
      auto it = std::find(openbps::materialfilter->bins_.begin(),
        openbps::materialfilter->bins_.end(), std::string(bin));
        if (it != openbps::materialfilter->bins_.end()){
            openbps::materialfilter->bins_.erase(it);
        }
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  return err;
}

extern "C" int
openbps_filters_get_bins_by_idx(int32_t index, char*** out)
{
  int err = 0;
  if (index >= 0 && index < openbps::filters.size()) {
    std::vector<int> tmp_indices; //clean indices in input (if Im not wrong)
    try {
      size_t i = 0;
      for (const auto& el : openbps::filters[index].bins_) {
        std::strcpy((*out)[i], el.c_str());
        i++;
      }
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  } else {
//    set_errmsg("Index in composition array is out of bounds.");
    return OPENBPS_E_OUT_OF_BOUNDS;
  }
  return err;
}

extern "C" int
openbps_filters_add_bin_by_idx (int32_t index, char* bin)
{
  int err = 0;
  if (index >= 0 && index < openbps::filters.size()) {
    std::vector<int> tmp_indices; //clean indices in input (if Im not wrong)
    try {
      openbps::filters[index].bins_.push_back({bin});
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  } else {
//    set_errmsg("Index in composition array is out of bounds.");
    return OPENBPS_E_OUT_OF_BOUNDS;
  }
  return err;
}

extern "C" int
openbps_filters_delete_bin_by_idx(int32_t index, char* bin)
{
  int err = 0;
  if (index >= 0 && index < openbps::filters.size()) {
    std::vector<int> tmp_indices; //clean indices in input (if Im not wrong)
    try {
        auto it = std::find(openbps::filters[index].bins_.begin(),
        openbps::filters[index].bins_.end(), std::string(bin));
        if (it != openbps::materialfilter->bins_.end()){
            openbps::materialfilter->bins_.erase(it);
        }
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  } else {
//    set_errmsg("Index in composition array is out of bounds.");
    return OPENBPS_E_OUT_OF_BOUNDS;
  }
  return err;
}

extern "C" int
openbps_time_filter_get_bins (double** out)
{
    int err = 0;
    try {
        *out = openbps::timefilter->bins_.data();
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  return err;
}

extern "C" int
openbps_time_filter_add_bin (double bin)
{
    int err = 0;
    try {
      openbps::timefilter->bins_.push_back(bin);
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  return err;
}

extern "C" int
openbps_time_filter_delete_bin(double bin)
{
    int err = 0;
    try {
      auto it = std::find(openbps::timefilter->bins_.begin(),
        openbps::timefilter->bins_.end(), bin);
        if (it != openbps::timefilter->bins_.end()){
            openbps::timefilter->bins_.erase(it);
        }
    } catch (const std::runtime_error& e) {
      return OPENBPS_E_DATA;
    }
  return err;
}