#include "openbps/reactions.h"

#include <string.h>

#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <vector>

#include "../extern/pugiData/pugixml.h"
#include "openbps/capi.h"
#include "openbps/configure.h"
#include "openbps/functionals.h"
#include "openbps/parse.h"

namespace openbps {

//==============================================================================
// Global variables
//==============================================================================

std::vector<std::unique_ptr<Composition>> compositions;
std::map<std::string, int> composmap;
int indexall{-1};
std::vector<Xslibs> externxslibs;

//==============================================================================
// Class cross-section implementations
//==============================================================================

Sxs::Sxs(pugi::xml_node node, const std::string& rxs,
         const std::string& redex) {
    this->xstype = node.attribute("reaction").value();
    this->xsname = node.attribute("name").value();
    if (rxs == "cs") {
        this->xs_ = get_node_array<udouble>(node, redex.c_str());
    } else {
        this->rxs = get_node_array<udouble>(node, redex.c_str());
    }
}

Xslibs::Xslibs(pugi::xml_node node) {
    // Get an energy multigroup structure
    if (check_for_node(node, "energy")) {
        for (pugi::xml_node tool : node.children("energy"))
            this->energies_ = get_node_array<double>(tool, "energy");
        this->numgroup = this->energies_.size();
    }
    if (check_for_node(node, "xslibs")) {
        parse_xml_xslibs_(node, this->xsdata);
    }
}

//==============================================================================
// Composition class implementation
//==============================================================================

Composition::Composition(pugi::xml_node node) {
    // Composition name
    if (check_for_node(node, "name")) {
        this->name_ = node.attribute("name").value();
    }
    // Read energy discretization
    if (check_for_node(node, "energy")) {
        for (pugi::xml_node tool : node.children("energy")) {
            size_t cng = std::stoi(tool.attribute("ng").value());
            this->energies_.insert(
                {cng, get_node_array<double>(tool, "energy")});
            this->energy_number_++;
        }
    }
    // Read a flux
    if (check_for_node(node, "flux")) {
        for (pugi::xml_node tool : node.children("flux")) {
            this->flux_ = get_node_array<udouble>(tool, "flux");
        }
    }
    if (check_for_node(node, "dflux")) {
        for (pugi::xml_node tool : node.children("dflux")) {
            std::vector<double> dflux{get_node_array<double>(tool, "dflux")};
            for (size_t j = 0; j < dflux.size() && j < flux_.size(); j++)
                flux_[j].Adddeviation(dflux[j]);
        }
    }
    // Read a spectrum (if a flux not presented)
    if (check_for_node(node, "spectrum")) {
        for (pugi::xml_node tool : node.children("spectrum")) {
            size_t cng = std::stoi(tool.attribute("ng").value());
            this->spectrum_ = get_node_array<udouble>(tool, "spectrum");
        }
    }

    if (check_for_node(node, "dspectrum")) {
        for (pugi::xml_node tool : node.children("dspectrum")) {
            std::vector<double> dspectrum{
                get_node_array<double>(tool, "dspectrum")};
            for (size_t j = 0; j < dspectrum.size() && j < spectrum_.size();
                 j++)
                spectrum_[j].Adddeviation(dspectrum[j]);
        }
    }
    // Read a cross-section data
    if (check_for_node(node, "xslibs")) parse_xml_xslibs_(node, this->xslib);

    this->spectrum_.size() > 0 ? this->energy_number_ = this->spectrum_.size()
                               : this->energy_number_ = this->flux_.size();
}

//! Auxilary function to copy data from xslib
Sxs parse_xs_xml_(pugi::xml_node node, const std::string& rxs,
                  const std::string& redex) {
    Sxs result(node, rxs, redex);
    return result;
}

//! Parse xslibs
void parse_xml_xslibs_(pugi::xml_node node, std::vector<Sxs>& xssource) {
    std::string rxs{node.child("xslibs").attribute("typex").value()};
    for (pugi::xml_node tool : node.child("xslibs").children("xslib")) {
        xssource.push_back(parse_xs_xml_(tool, rxs, "xslib"));
    }
    for (pugi::xml_node tool : node.child("xslibs").children("dxslib")) {
        // Read deviation part of cross section
        Sxs deriv_rxs(tool, rxs, "dxslib");
        for (auto ixs = xssource.begin(); ixs != xssource.end(); ixs++) {
            if (ixs->xsname == deriv_rxs.xsname &&
                ixs->xstype == deriv_rxs.xstype) {
                for (size_t j = 0;
                     j < deriv_rxs.rxs.size() && j < ixs->rxs.size(); j++)
                    ixs->rxs[j].Adddeviation(deriv_rxs.rxs[j]);
                for (size_t j = 0;
                     j < deriv_rxs.xs_.size() && j < ixs->xs_.size(); j++)
                    ixs->xs_[j].Adddeviation(deriv_rxs.xs_[j]);
            }
        }
    }
}

//! Auxilary function to copy data from xslib
void Composition::depcopymap_(std::map<size_t, std::vector<double>>& fmap,
                              std::map<size_t, std::vector<double>>& smap) {
    std::map<size_t, std::vector<double>>::iterator it;
    if (!smap.empty()) {
        for (it = smap.begin(); it != smap.end(); ++it) {
            if (fmap.find(it->first) == fmap.end()) {
                fmap[it->first] = it->second;
            }
        }
    }
}

//! Copy data from composition marked name "all" in xml file
void Composition::deploy_all(Composition& externcompos) {
    if (externcompos.name_ == this->name_) {
        return;
    } else {
        this->depcopymap_(this->energies_, externcompos.energies_);
        if (!externcompos.spectrum_.empty() && this->spectrum_.empty()) {
            this->spectrum_.resize(externcompos.spectrum_.size());
            std::copy(externcompos.spectrum_.begin(),
                      externcompos.spectrum_.end(), this->spectrum_.begin());
        }
        if (!externcompos.flux_.empty() && this->flux_.empty()) {
            this->flux_.resize(externcompos.flux_.size());
            std::copy(externcompos.flux_.begin(), externcompos.flux_.end(),
                      this->flux_.begin());
        }
        if (!externcompos.xslib.empty() && this->xslib.empty()) {
            this->xslib.resize(externcompos.xslib.size());
            std::copy(externcompos.xslib.begin(), externcompos.xslib.end(),
                      this->xslib.begin());
        }
    }
}

//! Calculate reaction-rate from cross-section data and flux
void Composition::calculate_rr_(Sxs& ixs, const std::vector<double>& extenergy,
                                udouble& rxs) {
    size_t ng = ixs.xs_.size();
    std::vector<udouble> cuflux(ng, 1.0);
    if (this->flux_.size() > 0) {
        if (this->flux_.size() == ng) {
            cuflux = this->flux_;
        } else {
            cuflux = collapsing<udouble>(this->energies_[this->flux_.size()],
                                         this->flux_, extenergy);
        }
    }
    for (int i = 0; i != ng; i++) rxs = rxs + cuflux[i] * ixs.xs_[i];
}

//! Calculate reaction rate for all reactions in xslib
void Composition::get_reaction() {
    for (auto ixs = this->xslib.begin(); ixs != this->xslib.end(); ixs++) {
        if (ixs->rxs.empty()) {
            ixs->rxs.resize(1);
            ixs->rxs[0] = 0.0;
            if (!ixs->xs_.empty()) {
                calculate_rr_(*ixs, this->energies_[ixs->xs_.size()],
                              ixs->rxs[0]);
            }  // if xs_ not empty
        }      // if rxs is empty
    }          // for
}

//! Import data from external cross section source
void Composition::import_xsdata(Xslibs& implibs) {
    for (auto& xs : implibs.xsdata) {
        if (std::find_if(xslib.begin(), xslib.end(), [&xs](Sxs& lib) {
                return xs.xsname == lib.xsname && xs.xstype == lib.xstype;
            }) == xslib.end()) {
            xslib.push_back(Sxs());
            xslib[xslib.size() - 1].xsname = xs.xsname;
            xslib[xslib.size() - 1].xstype = xs.xstype;
            xslib[xslib.size() - 1].rxs.resize(1);
            calculate_rr_(xs, implibs.get_egroups(),
                          xslib[xslib.size() - 1].rxs[0]);
        }
    }
}

//! Comparator function
bool compfe(std::pair<double, double> a, std::pair<double, double> b) {
    return a.first < b.first;
}

std::pair<std::vector<double>, std::vector<double>>
Composition::get_fluxenergy() {
    std::pair<std::vector<double>, std::vector<double>> result;
    size_t ng;
    double fluxnorm{0.0};
    if (spectrum_.empty()) {
        std::for_each(flux_.begin(), flux_.end(),
                      [&](udouble n) { fluxnorm += n.Real(); });
        for (auto& f : flux_) {
            spectrum_.push_back(f / fluxnorm);
        }
    }
    ng = spectrum_.size();

    result = std::make_pair(energies_[ng], usplit<double>(spectrum_).first);
    return result;
}

//==============================================================================
// Non class methods implementation
//==============================================================================
//! Read compositions from xml file
void read_reactions_xml() {
    using namespace configure;
    pugi::xml_document doc;
    auto result = doc.load_file(reaction_file.c_str());
    if (!result) {
        std::cout << "Warning: file reactions.xml not found!" << std::endl;
        // If reactions.xml not found run in decay only mode
        configure::decay_extra_out = true;
        return;
    }
    pugi::xml_node root_node = doc.child("compositions");
    if (configure::verbose)
        std::cout << "I' m in reactions.xml parser" << std::endl;

    for (pugi::xml_node tool : root_node.children("composit")) {
        compositions.push_back(
            std::unique_ptr<Composition>(new Composition(tool)));
        int index = compositions.size() - 1;
        if (compositions[index]->Name() == "all") {
            indexall = index;
        }
        composmap.insert({compositions[index]->Name(), index});
    }

    // Read imported cross-section libraries
    if (libs.size() > 0) read_importedlib_xml();

    for (size_t i = 0; i < compositions.size(); i++)
        if (i != indexall) {
            if (indexall > -1)
                compositions[i]->deploy_all(*compositions[indexall]);
            compositions[i]->get_reaction();
            if (externxslibs.size() > 0)
                for (auto& v : externxslibs) compositions[i]->import_xsdata(v);
        }
}

//! Read an imported cross section data library from *.xml files
void read_importedlib_xml() {
    for (auto& v : configure::libs) {
        pugi::xml_document doc;
        auto result = doc.load_file(v.c_str());
        if (!result) {
            std::cout << "Warning: file " << v << " not found!" << std::endl;
            return;
        }
        pugi::xml_node root_node = doc.child("importlib");
        externxslibs.emplace_back(root_node);
    }
}

}  // namespace openbps

//==============================================================================
// C API
//==============================================================================

extern "C" int openbps_composition_get_reaction(int32_t index) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            openbps::compositions[index]->get_reaction();
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_composition_get_fluxenergy(int32_t index, double** first,
                                                  double** second) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            auto res = openbps::compositions[index]->get_fluxenergy();
            *first = res.first.data();
            *second = res.second.data();
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_composition_deploy_all(int32_t compos_idx,
                                              int32_t extern_compos_idx) {
    int err = 0;
    if (compos_idx >= 0 && compos_idx < openbps::compositions.size() &&
        extern_compos_idx >= 0 &&
        extern_compos_idx < openbps::compositions.size()) {
        try {
            openbps::compositions[compos_idx]->deploy_all(
                *openbps::compositions[extern_compos_idx]);

        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_material_read_importedlib_xml() {
    try {
        openbps::read_importedlib_xml();
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }
    return 0;
}

extern "C" int openbps_material_read_reactions_xml() {
    try {
        openbps::read_reactions_xml();
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }
    return 0;
}

extern "C" int openbps_composition_import_xsdata(int32_t compos_idx,
                                                 int32_t implibs_idx) {
    int err = 0;
    if (compos_idx >= 0 && compos_idx < openbps::compositions.size() &&
        implibs_idx >= 0 && implibs_idx < openbps::externxslibs.size()) {
        try {
            openbps::compositions[compos_idx]->import_xsdata(
                openbps::externxslibs[implibs_idx]);

        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_add_xslibs(size_t numgroup, double* energies,
                                  size_t energies_len) {
    int err = 0;
    try {
        openbps::externxslibs.push_back(openbps::Xslibs());
        openbps::externxslibs.back().set_egroups(
            {energies, energies + energies_len});
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }
    return err;
}

extern "C" int openbps_get_egroups_by_index(int32_t index, double** energies) {
    int err = 0;
    if (index >= 0 && index < openbps::externxslibs.size()) {
        try {
            *energies = openbps::externxslibs[index].get_egroups().data();
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_delete_egroups_by_index(int32_t index,
                                               size_t energies_idx) {
    int err = 0;
    if (index >= 0 && index < openbps::externxslibs.size()) {
        try {
            openbps::externxslibs[index].delete_from_egroups(energies_idx);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_addto_egroups_by_index(int32_t index, double d) {
    int err = 0;
    if (index >= 0 && index < openbps::externxslibs.size()) {
        try {
            openbps::externxslibs[index].add_to_egroups(d);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

// extern "C" int openbps_append_sxs_to_xsdata(int32_t index, char* xsname,
// char* xstype) {
//     int err = 0;
//     if (index >= 0 && index < openbps::externxslibs.size()) {
//         try {
//             openbps::externxslibs[index].add_to_egroups(d);
//         } catch (const std::runtime_error& e) {
//             return OPENBPS_E_DATA;
//         }
//     } else {
//         //    set_errmsg("Index in composition array is out of bounds.");
//         return OPENBPS_E_OUT_OF_BOUNDS;
//     }
//     return err;
// }

//+
extern "C" int openbps_get_compsition_size(size_t* size) {
    int err = 0;
    try {
        *size = openbps::compositions.size();
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }

    return err;
}

extern "C" int openbps_get_xslibs_size_by_index(int32_t index, size_t* size) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            *size = openbps::compositions[index]->xslib.size();
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_add_composition(char* name, size_t nuclide_n,
                                       size_t energy_n) {
    int err = 0;
    try {
        openbps::compositions.push_back(std::make_unique<openbps::Composition>(
            std::string(name), nuclide_n, energy_n));
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }
    return err;
}

extern "C" int openbps_delete_composition_by_idx(int32_t index) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            openbps::compositions.erase(openbps::compositions.begin() + index);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_get_composition_data(int32_t index, char** name,
                                            size_t* nuclide_n,
                                            size_t* energy_n) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            std::cout << "I'm here!" << std::endl;
            //strcpy(*name, openbps::compositions[index]->Name().c_str());
            *name = const_cast<char*>(openbps::compositions[index]->Name().c_str());
            *nuclide_n = openbps::compositions[index]->NuclidNumber();
            *energy_n = openbps::compositions[index]->EnergyNumber();
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

//M
extern "C" int openbps_composition_get_all_keys_energy(int32_t index, int** res, int* pSize) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            auto keys = openbps::compositions[index]->get_all_keys();
			int len = keys.size();
            if (*pSize != len) {
                if (*pSize == 0) {
                    *pSize = len;
                    *res = (int*)malloc(*pSize);
                } else {
                    *pSize = len;
                    *res = (int*)malloc(*pSize);
                }
            }
            for (size_t i = 0; i < keys.size(); i++)
                (*res)[i] = keys[i];
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_composition_get_energy_by_key(int32_t index, size_t key, double** res) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            *res = openbps::compositions[index]->get_energy(key).data();
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_composition_set_energy(int32_t index, size_t key, double* en, size_t en_size) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            openbps::compositions[index]->set_energy({en, en + en_size}, key);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

//M
extern "C" int openbps_composition_get_spectrum(int32_t index, double** s_real, double** s_dev, int* pSize) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            auto sp = openbps::compositions[index]->get_spectrum();
			int len = sp.size();
            if (*pSize != len) {
                if (*pSize == 0) {
                    *pSize = len;
                    *s_real = (double*)malloc(*pSize);
					*s_dev = (double*)malloc(*pSize);
                } else {
                    *pSize = len;
                    *s_real = (double*)malloc(*pSize);
					*s_dev = (double*)malloc(*pSize);
                }
            }
            for (size_t i = 0; i < sp.size(); i++){
               (*s_real)[i] = sp[i].Real();
               (*s_dev)[i] = sp[i].Dev();
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

extern "C" int openbps_composition_add_to_spectrum(int32_t index, double s_real, double s_dev) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            openbps::compositions[index]->add_to_spectrum(openbps::udouble(s_real, s_dev));
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

//M
extern "C" int openbps_get_xslib_elem_by_index(
    int32_t index, size_t xlib_idx, char** name, char** type, double** real_rxs,
    double** dev_rxs, double** real_xs, double** dev_xs, int* rxs_size, int* xs_size) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            //strcpy(
            //    *name,
            //    openbps::compositions[index]->xslib[xlib_idx].xsname.c_str());
            *name = const_cast<char*>(openbps::compositions[index]->xslib[xlib_idx].xsname.c_str());
            //strcpy(
            //    *type,
            //    openbps::compositions[index]->xslib[xlib_idx].xstype.c_str());
            *type = const_cast<char*>(openbps::compositions[index]->xslib[xlib_idx].xstype.c_str());
            size_t i = 0;
			if (openbps::compositions[index]->xslib[xlib_idx].xstype == "cs") {
				for (auto& el : openbps::compositions[index]->xslib[xlib_idx].xs_) {
                    (*real_xs)[i] = el.Real();
                    (*dev_xs)[i] = el.Dev();
					i++;
                }
			} else {
				for (auto& el : openbps::compositions[index]->xslib[xlib_idx].rxs) {
                    (*real_rxs)[i] = el.Real();
                    (*dev_rxs)[i] = el.Dev();
					i++;
                }
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

//M
extern "C" int openbps_add_xslib_elem(int32_t index, char* name, char* type,
                                      double* real_rxs, double* dev_rxs,
                                      int32_t rxs_size, double* real_xs,
                                      double* dev_xs, int32_t xs_size) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            openbps::Sxs sxs;
            sxs.xsname = {name};
            sxs.xstype = {type};
			if (sxs.xstype == "cs") {
				for (size_t i = 0; i < xs_size; i++) {
                    sxs.xs_.push_back(openbps::udouble(real_xs[i], dev_xs[i]));
                }
			} else {
                for (size_t i = 0; i < rxs_size; i++) {
                     sxs.rxs.push_back(openbps::udouble(real_rxs[i], dev_rxs[i]));
                }
			}
            openbps::compositions[index]->xslib.push_back(sxs);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_delete_xslib_elem(int32_t index, size_t xlib_idx) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            openbps::compositions[index]->xslib.erase(
                openbps::compositions[index]->xslib.begin() + xlib_idx);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}
//-----------------------------------------------------------------------------
extern "C" int openbps_composition_delete_energy(int32_t index, size_t key, double* en, size_t en_size) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            openbps::compositions[index]->delete_energy(key);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_composition_get_flux(int32_t index, double** f_real, double** f_dev) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            auto fl = openbps::compositions[index]->get_flux();
            for (size_t i = 0; i < fl.size(); i++){
               *f_real[i] = fl[i].Real();
               *f_dev[i] = fl[i].Dev();
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

extern "C" int openbps_composition_add_to_flux(int32_t index, double f_real, double f_dev) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            openbps::compositions[index]->add_to_flux(openbps::udouble(f_real, f_dev));
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_composition_delete_from_spectrum(int32_t index, size_t pos) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            if (pos < openbps::compositions[index]->get_spectrum().size())
                openbps::compositions[index]->delete_from_spectrum(pos);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_composition_delete_from_flux(int32_t index, size_t pos) {
    int err = 0;
    if (index >= 0 && index < openbps::compositions.size()) {
        try {
            if (pos < openbps::compositions[index]->get_flux().size())
                openbps::compositions[index]->delete_from_flux(pos);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in composition array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}