#include "openbps/materials.h"
#include "openbps/configure.h"
#include "openbps/parse.h"
#include "openbps/nuclide.h"
#include <cstring>
#include <memory>
#include <stdlib.h>
#include <algorithm>
#include "../extern/pugiData/pugixml.h"
#include "openbps/reactions.h"
#include "openbps/capi.h"

namespace openbps {

std::vector<std::unique_ptr<Materials>> materials; //!< vector consisting
                                                   //!< all materials

//==============================================================================
// Materials class implementation
//==============================================================================
Materials::Materials(){}

void Materials::xml_add_material(pugi::xml_node node) {
    auto material = node.append_child("material");
    material.append_attribute("name") =  this->Name().c_str();
    material.append_attribute("volume") =this->Volume();
    material.append_attribute("mass") =  this->Mass();
    material.append_attribute("power") = this->Power();

    auto nameofn = material.append_child("namenuclides");
    nameofn.append_child(pugi::node_pcdata).set_value(join(namenuclides,
                                                           " ").c_str());
    auto concpair = usplit(this->conc);
    auto concnod = material.append_child("conc");
    concnod.append_child(pugi::node_pcdata).set_value(joinDouble(concpair.first,
                                                                 " ").c_str());
    auto dconcnod = material.append_child("dconc");
    dconcnod.append_child(pugi::node_pcdata).set_value(joinDouble(concpair.second,
                                                                  " ").c_str());
}

void Materials::add_nuclide(const std::string& extname, udouble extconc) {
    auto it = std::find(this->namenuclides.begin(),
                        this->namenuclides.end(), extname);
    if (it == this->namenuclides.end()) {
        this->namenuclides.push_back(extname);
        this->conc.push_back(extconc);
        this->indexnuclides.push_back(get_nuclidarray_index(extname));
    } else {
        auto index = std::distance(this->namenuclides.begin(), it);
        this->conc[index] = extconc;
    }
}

void Materials::delete_nuclide(const std::string& extname) {
    auto it = std::find(this->namenuclides.begin(),
                        this->namenuclides.end(), extname);
    if (it != this->namenuclides.end()) {
        auto index = std::distance(this->namenuclides.begin(), it);
        this->namenuclides.erase(it);
        this->conc.erase(conc.begin() + index);
        this->indexnuclides.erase(indexnuclides.begin() + index);
    }
}

/*void Materials::bindcomposition(Composition& extcompos){
  this->compos = std::make_shared<Composition>(extcompos);
}*/

//==============================================================================
// Non class method implementation
//==============================================================================

//! Write down materials vector information into xml file
void form_materials_xml(std::string xml_path) {
    pugi::xml_document doc;
    auto declarationNode = doc.append_child(pugi::node_declaration);
    declarationNode.append_attribute("version") = "1.0";
    declarationNode.append_attribute("encoding") = "UTF-8";

    auto materialsnod = doc.append_child("materials");
    for(int i = 0; i < materials.size(); i++) {
        materials[i]->xml_add_material(materialsnod);
        if (configure::verbose)
            std :: cout << "material was included" << std ::endl;
    }
    bool saveSucceeded = doc.save_file(xml_path.c_str(), PUGIXML_TEXT("  "));
}

//! Read materials from xml file to a vector
void read_materials_from_inp(std::string inp_path
                             ) {
    pugi::xml_document doc;
    // Load a file
	auto result = doc.load_file(inp_path.c_str());
	if (!result) {
	    std::cerr << "Error: file not found!" << std::endl;
	}
	pugi::xml_node root_node = doc.child("materials");
    // Iterate over materials records
	for (pugi::xml_node tool : root_node.children("material")) {
        std::string mname {tool.attribute("name").value()};
        auto m = new Materials(mname,
                    atof(tool.attribute("volume").value()),
                    atof(tool.attribute("volume").value()),
                    atof(tool.attribute("power").value()));
        m->namenuclides = get_node_array<std::string>(tool, "namenuclides");
        std::for_each(m->namenuclides.begin(), m->namenuclides.end(),
                      [m](std::string& nucname) {
                        m->indexnuclides.push_back(get_nuclidarray_index(
                                                      nucname));
                      }
        );
        // Read a concentration
        m->conc = get_node_array<udouble>(tool, "conc");
        if (check_for_node(tool, "dconc")) {
            std::vector<double> dconc = get_node_array<double>(tool, "dconc");
            int j = 0;
            if (dconc.size() <= m->conc.size())
                std::for_each(dconc.begin(), dconc.end(), [&j, m](double& dc) {
                    m->conc[j].Adddeviation(dc);
                    j++; }
                );
        }
        materials.push_back(std::unique_ptr<Materials>(m));
    } // for material
    return;
}

//! Look up all compositions and find a match with material name
//!
void matchcompositions() {
    for (auto&  mat: materials) {
        for  (size_t j = 0; j < compositions.size(); j++ ) {
            if (mat->Name()==compositions[j]->Name()) {
                mat->numcomposition = j;
                break;
            }
        }
    }
}

} //namespace openbps

//==============================================================================
// C API
//==============================================================================

extern "C" int openbps_material_add_nuclide(int32_t index, const char* extname,
                                            double real, double dev) {
    int err = 0;
    if (index >= 0 && index < openbps::materials.size()) {
        try {
            openbps::materials[index]->add_nuclide(extname, {real, dev});
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in materials array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_material_delete_nuclide(int32_t index,
                                               const char* extname) {
    int err = 0;
    if (index >= 0 && index < openbps::materials.size()) {
        try {
            openbps::materials[index]->delete_nuclide(extname);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in materials array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_material_matchcompositions() {
    try {
        openbps::matchcompositions();
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }
    return 0;
}

extern "C" int openbps_read_materials_from_inp(char* inp_path) {
    try {
        openbps::read_materials_from_inp({inp_path});
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }
    return 0;
}

extern "C" int openbps_form_materials_xml(char* inp_path) {
    try {
        openbps::form_materials_xml({inp_path});
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }
    return 0;
}

extern "C" int openbps_material_add(char* name, double volume, double power,
                                    double mass) {
    int err = 0;
    try {
        openbps::materials.push_back(std::make_unique<openbps::Materials>(
            std::string(name), volume, power, mass));
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }
    return err;
}

extern "C" int openbps_material_set_params_by_idx(int32_t index, char* name,
                                                  double volume, double power,
                                                  double mass) {
    int err = 0;
    if (index >= 0 && index < openbps::materials.size()) {
        try {
            openbps::materials[index]->setMass(mass);
            openbps::materials[index]->setName({name});
            openbps::materials[index]->setVolume(volume);
            openbps::materials[index]->setPower(power);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in materials array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_material_delete_by_idx(int32_t index) {
    int err = 0;
    if (index >= 0 && index < openbps::materials.size()) {
        try {
            openbps::materials.erase(openbps::materials.begin() + index);
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in materials array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_materials_get_size(size_t* s) {
    try {
        *s = openbps::materials.size();
    } catch (const std::runtime_error& e) {
        return OPENBPS_E_DATA;
    }
    return 0;
}

extern "C" int openbps_material_get_params_by_idx(int32_t index, char** name,
                                                  double* mass, double* volume,
                                                  double* power) {
    int err = 0;
    if (index >= 0 && index < openbps::materials.size()) {
        try {
            *name = openbps::materials[index]->Name().data();
            *mass = openbps::materials[index]->Mass();
            *volume = openbps::materials[index]->Volume();
            *power = openbps::materials[index]->Power();
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in materials array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_material_set_nuclides_conc(int32_t index, char* name,
                                                  double realconc,
                                                  double devconc) {
    int err = 0;
    if (index >= 0 && index < openbps::materials.size()) {
        try {
            std::string extname {name};
            auto it = std::find(openbps::materials[index]->namenuclides.begin(),
                        openbps::materials[index]->namenuclides.end(), extname);
            if (it != openbps::materials[index]->namenuclides.end()) {
                //auto index = std::distance(openbps::materials[index]->namenuclides.begin(), it);
                openbps::materials[index]->conc[it - openbps::materials[index]->namenuclides.begin()] = {realconc, devconc};
            }

        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in materials array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_material_get_nuclides_by_idx(int32_t index,
                                                    char*** nuclides,
                                                    int* pSize) {
    int err = 0;
    if (index >= 0 && index < openbps::materials.size()) {
        try {
            size_t i = 0;
            int len = openbps::materials[index]->namenuclides.size();
            if (*pSize != len) {
                if (*pSize == 0) {
                    *pSize = len;
                    *nuclides = (char**)malloc(*pSize);
                } else {
                    *pSize = len;
                    *nuclides = (char**)realloc(*nuclides, *pSize);
                }
            }
            for (auto& el : openbps::materials[index]->namenuclides) {
                (*nuclides)[i] = const_cast<char*>(el.c_str());
                i++;
            }

        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {
        //    set_errmsg("Index in materials array is out of bounds.");
        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_material_get_conc_by_idx(int32_t index, double** real,
                                                               double** dev,
                                                               int* pSize) {
    int err = 0;
    if (index >= 0 && index < openbps::materials.size()) {
        try {
            size_t i = 0;
            int len = openbps::materials[index]->namenuclides.size();
            if (*pSize != len) {
                if (*pSize == 0) {
                    *pSize = len;
                    *real = (double*)malloc(*pSize);
                    *dev = (double*)malloc(*pSize);
                } else {
                    *pSize = len;
                    *real = (double*)realloc(*real, *pSize);
                    *dev = (double*)realloc(*dev, *pSize);
                }
            }
            for (auto& el : openbps::materials[index]->conc) {
                (*real)[i] = el.Real();
                (*dev)[i] = el.Dev();
                i++;
            }
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {

        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}

extern "C" int openbps_material_get_idx_nuclides_by_idx(int32_t index,
                                                        int** index_nuclides,
                                                        int* pSize) {
    int err = 0;

    if (index >= 0 && index < openbps::materials.size()) {
        try {
            int len = openbps::materials[index]->namenuclides.size();
            if (*pSize != len) {
                if (*pSize == 0) {
                    *pSize = len;
                    *index_nuclides = (int*)malloc(*pSize);
                } else {
                    *pSize = len;
                    *index_nuclides = (int*)realloc(*index_nuclides, *pSize);
                }
            }
            for (int i = 0; i < *pSize; i++) {
                (*index_nuclides)[i] = openbps::materials[index]->indexnuclides[i];
            }
        } catch (const std::runtime_error& e) {
            return OPENBPS_E_DATA;
        }
    } else {

        return OPENBPS_E_OUT_OF_BOUNDS;
    }
    return err;
}