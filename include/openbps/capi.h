#ifndef OPENPbps_CAPI_H
#define OPENBPS_CAPI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

// materials
int openbps_material_add_nuclide(int32_t index, const char *extname,
                                 double real, double dev);
int openbps_material_delete_nuclide(int32_t index, const char *extname);
int openbps_material_matchcompositions();
int openbps_read_materials_from_inp(char *inp_path);
int openbps_form_materials_xml(char *inp_path);
int openbps_material_delete_by_idx(int32_t index);
int openbps_material_set_params_by_idx(int32_t index, char *name, double volume,
                                       double power, double mass);
int openbps_material_add(char* name, double volume, double power, double mass);
int openbps_materials_get_size(size_t *s);
int openbps_material_get_params_by_idx(int32_t index, char **name, double *mass,
                                       double *volume, double *power);
int openbps_material_set_nuclides_conc(int32_t index, char* name,double realconc,double devconc);
int openbps_material_get_idx_nuclides_by_idx(int32_t index, int** index_nuclides,int* pSize);
int openbps_material_get_conc_by_idx(int32_t index, double** real, double** dev,int* pSize);
int openbps_material_get_nuclides_by_idx(int32_t index, char*** nuclides,
                                         int* pSize);

// reactions
int openbps_composition_get_reaction(int32_t index);
int openbps_composition_get_fluxenergy(int32_t index, double **first,
                                       double **second);
int openbps_composition_deploy_all(int32_t compos_idx,
                                   int32_t extern_compos_idx);
int openbps_material_read_importedlib_xml();
int openbps_material_read_reactions_xml();
int openbps_composition_deploy_all(int32_t compos_idx,
                                   int32_t extern_compos_idx);
int openbps_composition_import_xsdata(int32_t compos_idx, int32_t implibps_idx);

// modify sxs vector in compositions
int openbps_get_xslibs_size_by_index(int32_t index, int *size);
int openbps_add_xslib_elem(int32_t index, char *name, char *type,
                           double *real_rxs, double *dev_rxs, int32_t rxs_size,
                           double *real_xs, double *dev_xs, int32_t xs_size);
int openbps_update_xslib_elem(int32_t index, char* name, char* type,
                              double* real_rxs, double* dev_rxs,
                              int32_t rxs_size, double* real_xs,
                              double* dev_xs, int32_t xs_size);
int openbps_get_xslib_elem_by_index(int32_t index, size_t xlib_idx,
                                    char** name, char** type,
									double** real_rxs, double** dev_rxs,
									double** real_xs, double** dev_xs,
									int* rxs_size, int* xs_size);
int openbps_delete_xslib_elem(int32_t index, size_t xlib_idx);

//compositions
int openbps_get_compsition_size(int* size);
int openbps_get_composition_data(int32_t index, char **name, size_t *nuclide_n,
                                 size_t *energy_n);
int openbps_add_composition(char *name, size_t nuclide_n, size_t energy_n);
int openbps_delete_composition_by_idx(int32_t index);
int openbps_composition_get_energy_by_key(int32_t index, size_t key,
                                          double **res, int* pSize);
int openbps_composition_set_energy(int32_t index, size_t key, double *en,
                                   size_t en_size);
int openbps_composition_delete_energy(int32_t index, size_t key, double *en,
                                      size_t en_size);
int openbps_composition_get_all_keys_energy(int32_t index, int** res, int* pSize);

//flux&spectrum

int openbps_composition_get_spectrum(int32_t index, double** s_real, double** s_dev, int* pSize);
int openbps_composition_get_flux(int32_t index, double** f_real, double** f_dev);
int openbps_composition_add_to_spectrum(int32_t index, double s_real, double s_dev);
int openbps_composition_add_to_flux(int32_t index, double f_real, double f_dev);
int openbps_composition_delete_from_spectrum(int32_t index, size_t pos);
int openbps_composition_delete_from_flux(int32_t index, size_t pos);

// filter
int openbps_filter_apply(int32_t index, const char **input, size_t input_size,
                         int **indices);
int openbps_material_filter_apply(const char *mat_name, bool *is_valid);
int openbps_time_filter_apply(double dt, int numstep, int **indices);
int openbps_material_filter_get_bins(char ***out);
int openbps_material_filter_add_bin(char *bin);
int openbps_material_filter_delete_bin(char *bin);
int openbps_filters_get_bins_by_idx(int32_t index, char ***out);
int openbps_filters_add_bin_by_idx(int32_t index, char *bin);
int openbps_filters_delete_bin_by_idx(int32_t index, char *bin);
int openbps_time_filter_get_bins(double **out);
int openbps_time_filter_add_bin(double bin);
int openbps_time_filter_delete_bin(double bin);

// nuclides
int openbps_get_nuclidearray_index(const char *name);
int openbps_read_nuclide_xml(const char *filepath);
int openbps_chain_nuclides_add(char *name, int Z, int A, int M, double awr);
int openbps_set_nuclide_name_idx(const char *name, size_t idx);
int openbps_get_nuclide_index(const char *name, size_t *idx);
int openbps_delete_nuclide_by_name(const char *name);
int openbps_chain_nuclides_get_name_by_index(int32_t index, char **name);
int openbps_chain_nuclides_set_name_by_index(int32_t index, char *name);
int openbps_chain_nuclides_get_Z_by_index(int32_t index, int *Z);
int openbps_chain_nuclides_set_Z_by_index(int32_t index, int Z);
int openbps_chain_nuclides_get_A_by_index(int32_t index, int *A);
int openbps_chain_nuclides_set_A_by_index(int32_t index, int A);
int openbps_chain_nuclides_get_m_by_index(int32_t index, int *m);
int openbps_chain_nuclides_set_m_by_index(int32_t index, int m);
int openbps_chain_nuclides_set_awr_by_index(int32_t index, double awr);
int openbps_chain_nuclides_get_awr_by_index(int32_t index, double *awr);
int openbps_chain_nuclides_add_set_hl_by_index(int32_t index, double real,
                                               double dev);
int openbps_chain_nuclides_get_hl_by_index(int32_t index, double *real,
                                           double *dev);
int openbps_chain_nuclides_set_decay_eng_by_index(int32_t index, double real,
                                                  double dev);
int openbps_chain_nuclides_get_decay_eng_by_index(int32_t index, double *real,
                                                  double *dev);

    extern int OPENBPS_E_UNASSIGNED;
    extern int OPENBPS_E_ALLOCATE;
    extern int OPENBPS_E_OUT_OF_BOUNDS;
    extern int OPENBPS_E_INVALID_SIZE;
    extern int OPENBPS_E_INVALID_ARGUMENT;
    extern int OPENBPS_E_INVALID_TYPE;
    extern int OPENBPS_E_INVALID_ID;
    extern int OPENBPS_E_GEOMETRY;
    extern int OPENBPS_E_DATA;
    extern int OPENBPS_E_PHYSICS;
    extern int OPENBPS_E_WARNING;

  // Global variables
    extern char OPENBPS_err_msg[256];

#ifdef __cplusplus
}
#endif

#endif // OPENBPS_CAPI_H