#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Fedorov
#
# Created:     18.03.2021
# Copyright:   (c) Fedorov 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from ctypes import CDLL, c_bool, c_int
from ctypes import *
from numpy.ctypeslib import as_array
import numpy as np
import os
import sys

if sys.platform == 'darwin':
    _suffix = 'dylib'
else:
    _suffix = 'so'

nameoflib = "/home/user/lad/OpenBPS/build/lib/liblibopenbps.{}"



LIB = CDLL(nameoflib.format(_suffix))
#
LIB.openbps_init.argtypes = [c_int, POINTER(POINTER(c_char))]
LIB.openbps_init.restype = c_int
#LIB.openbps_init.errcheck = _error_handler
#
LIB.openbps_run.restype = c_int
LIB.openbps_finalize.restype = c_int
#
LIB.openbps_material_add_nuclide.restype = c_int
LIB.openbps_material_add_nuclide.argtypes=[c_int, c_char_p, c_double, c_double]
LIB.openbps_material_add.restype = c_int
LIB.openbps_material_add.argtypes=[c_char_p, c_double, c_double, c_double]
LIB.openbps_material_delete_nuclide.restype = c_int
LIB.openbps_material_delete_nuclide.argtypes=[c_int, c_char_p]
LIB.openbps_material_delete_by_idx.restype = c_int
LIB.openbps_material_delete_by_idx.argtypes=[c_int]
LIB.openbps_material_set_params_by_idx.restype = c_int
LIB.openbps_material_set_params_by_idx.argtypes=[c_int, c_char_p, c_double,
                                                 c_double, c_double]
LIB.openbps_material_set_nuclides_conc.restype = c_int
LIB.openbps_material_set_nuclides_conc.argtypes=[c_int, c_char_p, c_double,
                                                 c_double]
LIB.openbps_materials_get_size.restype = c_int
LIB.openbps_materials_get_size.argtypes=[POINTER(c_int)]
LIB.openbps_material_get_params_by_idx.restype = c_int
LIB.openbps_material_get_params_by_idx.argtypes=[c_int, POINTER(c_char_p),
                                                 POINTER(c_double),
                                                 POINTER(c_double),
                                                 POINTER(c_double)]
LIB.openbps_material_get_idx_nuclides_by_idx.restype = c_int
LIB.openbps_material_get_idx_nuclides_by_idx.argtypes=[c_int,
                                                       POINTER(POINTER(c_int)),
                                                       POINTER(c_int)]
LIB.openbps_material_get_conc_by_idx.restype = c_int
LIB.openbps_material_get_conc_by_idx.argtypes=[c_int,
                                               POINTER(POINTER(c_double)),
                                               POINTER(POINTER(c_double)),
                                               POINTER(c_int)]

LIB.openbps_material_get_nuclides_by_idx.restype = c_int
LIB.openbps_material_get_nuclides_by_idx.argtypes=[c_int,
                                                  POINTER(POINTER(c_char_p)),
                                                  POINTER(c_int)]
#REACTIONS
LIB.openbps_get_compsition_size.restype = c_int
LIB.openbps_get_compsition_size.argtypes=[POINTER(c_int)]
LIB.openbps_get_xslibs_size_by_index.restype = c_int
LIB.openbps_get_xslibs_size_by_index.argtypes=[c_int, POINTER(c_int)]
LIB.openbps_add_composition.restype = c_int
LIB.openbps_add_composition.argtypes=[c_char_p, c_int, c_int]

LIB.openbps_delete_composition_by_idx.restype = c_int
LIB.openbps_delete_composition_by_idx.argtypes=[c_int]
LIB.openbps_get_composition_data.restype = c_int
LIB.openbps_get_composition_data.argtypes=[c_int, POINTER(c_char_p),
                                           POINTER(c_int), POINTER(c_int)]
LIB.openbps_composition_get_all_keys_energy.restype = c_int
LIB.openbps_composition_get_all_keys_energy.argtypes=[c_int, POINTER(POINTER(c_int)), POINTER(c_int)]
LIB.openbps_composition_get_energy_by_key.restype = c_int
LIB.openbps_composition_get_energy_by_key.argtypes=[c_int, c_int, POINTER(POINTER(c_double)), POINTER(c_int)]
LIB.openbps_composition_set_energy.restype = c_int
LIB.openbps_composition_set_energy.argtypes=[c_int, c_int, POINTER(c_double), c_int]
LIB.openbps_composition_get_spectrum.restype = c_int
LIB.openbps_composition_get_spectrum.argtypes=[c_int, POINTER(POINTER(c_double)),
                                               POINTER(POINTER(c_double)),POINTER(c_int)]
LIB.openbps_composition_add_to_spectrum.restype = c_int
LIB.openbps_composition_add_to_spectrum.argtypes=[c_int, c_double, c_double]
LIB.openbps_get_xslib_elem_by_index.restype = c_int
LIB.openbps_get_xslib_elem_by_index.argtypes=[c_int, c_int, POINTER(c_char_p),
                                              POINTER(c_char_p), POINTER(POINTER(c_double)),
                                              POINTER(POINTER(c_double)), POINTER(POINTER(c_double)),
                                              POINTER(POINTER(c_double)), POINTER(c_int),
                                              POINTER(c_int)]

LIB.openbps_add_xslib_elem.restype = c_int
LIB.openbps_add_xslib_elem.argtypes=[c_int, c_char_p, c_char_p,
                                     POINTER(c_double), POINTER(c_double),
                                     c_int, POINTER(c_double), POINTER(c_double),
                                     c_int]

LIB.openbps_update_xslib_elem.restype = c_int
LIB.openbps_update_xslib_elem.argtypes=[c_int, c_char_p, c_char_p,
                                     POINTER(c_double), POINTER(c_double),
                                     c_int, POINTER(c_double), POINTER(c_double),
                                     c_int]
def finalize():
    """Finalize simulation and free memory"""
    LIB.openbps_finalize()

def init(args=None):
    """Initialize OpenMC
    Parameters
    ----------
    args : list of str
        Command-line arguments
    """
    if args is not None:
        args = [''] + list(args)
    else:
        args = ['']

    argc = len(args)
    # Create the argv array. Note that it is actually expected to be of
    # length argc + 1 with the final item being a null pointer.
    argv = (POINTER(c_char) * (argc + 1))()
    for i, arg in enumerate(args):
        argv[i] = create_string_buffer(arg.encode())

    LIB.openbps_init(argc, argv)

def run():
    """Finalize simulation and free memory"""
    LIB.openbps_run()

class _FortranObject:
    def __repr__(self):
        return "{}[{}]".format(type(self).__name__, self._index)


class _FortranObjectWithID(_FortranObject):
    def __init__(self, uid=None, new=True, index=None):
        # Creating the object has already been handled by __new__. In the
        # initializer, all we do is make sure that the object returned has an ID
        # assigned. If the array index of the object is out of bounds, an
        # OutOfBoundsError will be raised here by virtue of referencing self.id
        self.id

class OpenBPSSxs:

    def __init__(self, index=-1,
                  name=None, xtype=None):
        self._initialize()
        if (index > -1):
            self._index = c_int(index)
            #self._read_material()
        else:
            self._index = index
            self._name = name
            self._xtype = xtype

    def _initialize(self):
        self._index = -1
        self._name = None
        self._xtype = None
        self._lenrx = c_int(0)
        self._lencs  = c_int(0)
        self._csr = POINTER(c_double)()
        self._csd = POINTER(c_double)()
        self._rxr = POINTER(c_double)()
        self._rxd = POINTER(c_double)()

    def create_cross_section_mode(self, xname, xtype, real, dev):
        self._xtype = xtype
        self._xname = xname
        real = np.asarray(real)
        dev = np.asarray(dev)
        self._lencs = c_int(len(real))
        self._cdr = real.ctypes.data_as(POINTER(c_double))
        self._csd = dev.ctypes.data_as(POINTER(c_double))

    def create_reactions_mode(self, xname, xtype, real, dev):
        self._xtype = xtype
        self._xname = xname
        real = np.asarray(real)
        dev = np.asarray(dev)
        print(real, dev)
        self._lenrx = c_int(len(real))
        self._rxr = real.ctypes.data_as(POINTER(c_double))
        self._rxd = dev.ctypes.data_as(POINTER(c_double))

    def get_by_composition(self, number):
        name = c_char_p()
        xtype = c_char_p()
        print("Number, index ", number, self._index)
        LIB.openbps_get_xslib_elem_by_index(c_int(number), self._index,
                                            name, xtype,
                                            self._rxr, self._rxd,
                                            self._csr, self._csd,
                                            self._lenrx, self._lencs)

        self._name = name.value.decode()
        self._xtype = xtype.value.decode()

    def add_from_composition(self, number):
        name_ptr = c_char_p(self._name.encode())
        xtype_ptr = c_char_p(self._xtype.encode())

        LIB.openbps_add_xslib_elem(c_int(number),
                                   name_ptr, xtype_ptr,
                                   self._rxr, self._rxd,
                                   self._lenrx,
                                   self._csr, self._csd,
                                   self._lencs)
    def update_from_composition(self, number):
        name_ptr = c_char_p(self._name.encode())
        xtype_ptr = c_char_p(self._xtype.encode())
        LIB.openbps_update_xslib_elem(c_int(number),
                                   name_ptr, xtype_ptr,
                                   self._rxr, self._rxd,
                                   self._lenrx,
                                   self._csr, self._csd,
                                   self._lencs)

    @staticmethod
    def get_size_by_composition(number):
        n = c_int()
        LIB.openbps_get_xslibs_size_by_index(c_int(number), n)
        return n.value

    @property
    def rx(self):
        return [(self._rxr[n], self._rxd[n])
                for n in range(self._lenrx.value)]
    @property
    def cs(self):
        return [(self._csr[n], self._csd[n])
                for n in range(self._lencs.value)]
    @property
    def name(self):
        return self._name

    @property
    def xtype(self):
        return self._xtype

class OpenBPSComposition:

    def __init__(self, index=-1, name=None,
                 numNuclid=None, numEnergy=None):
        self._initialize()
        if (index > -1):
            self._index = index
            self._read_composition()
        else:
            self._name = name
            self._index = index
            self._numNuclid = numNuclid
            self._numEnergy = numEnergy

    def _initialize(self):
        self._index = -1
        self._name = None
        self._numNuclid = None
        self._numEnergy = None
        self._lenenergymap = c_int(0)
        self._lenspectrum  = c_int(0)
        self._energies = {}
        self._energykeys = POINTER(c_int)()
        self._spectrumr = POINTER(c_double)()
        self._spectrumd = POINTER(c_double)()
        self._Sxs = []
    def _read_composition(self):
        name = c_char_p()
        numNuclid = c_int()
        numEnergy = c_int()
        LIB.openbps_get_composition_data(self._index, name,
                                         numNuclid, numEnergy)
        self._name = name.value.decode()
        self._numNuclid = numNuclid.value
        self._numEnergy = numEnergy.value

    @staticmethod
    def get_size_composition():
        n = c_int()
        LIB.openbps_get_compsition_size(n)
        return n.value

    @staticmethod
    def add_composition(name, numNuclid, numEnergy):
        name_ptr = c_char_p(name.encode())
        numNuclid = c_int(numNuclid)
        numEnergy = c_int(numEnergy)
        LIB.openbps_add_composition(name_ptr, numNuclid, numEnergy)

    def del_composition(self):

        LIB.openbps_delete_composition_by_idx(self._index)

    def _get_energy_keys(self):
        # Allocate memory for arguments that are written to
        LIB.openbps_composition_get_all_keys_energy(self._index,
                                                     self._energykeys,
                                                     self._lenenergymap)
    def _get_energies(self, key, ptr, energies):
        LIB.openbps_composition_get_energy_by_key(self._index,
                                                  c_int(key),
                                                  energies,
                                                  ptr)
        print(energies[0])

    def _get_spectrum(self):
        print(LIB.openbps_composition_get_spectrum(self._index, self._spectrumr,
                                             self._spectrumd,
                                             self._lenspectrum))
        print('a')
        print(self._lenspectrum)

    def read_energies(self):
        self._get_energy_keys()
        for i in range(self._lenenergymap.value):
            k = self._energykeys[i]
            self._energies[k] = (c_int(0), POINTER(c_double)())
            self._get_energies(k, self._energies[k][0], self._energies[k][1])


    def write_energies(self, energies):
        key = c_int(len(energies) - 1)
        size = c_int(len(energies))
        # Get numpy array as a double*
        e = np.asarray(energies)
        ep = e.ctypes.data_as(POINTER(c_double))
        LIB.openbps_composition_set_energy(self._index,
                                           key, ep, key)
    def write_spectrum(self, spectrumr, spectrumd):
        for sr, sd in zip(spectrumr, spectrumd):
            LIB.openbps_composition_add_to_spectrum(self._index,
                                                    c_double(sr), c_double(0.0))

    def getSxs(self):
        num = OpenBPSSxs.get_size_by_composition(self._index)
        print("Xslib size ", num)
        for n in range(num):
            _sxs = OpenBPSSxs(n)
            _sxs.get_by_composition(self._index)
            self._Sxs.append(_sxs)

    def writeSxs(self, xname, xtype, real, dev, sectype="rx"):
        _sxs = OpenBPSSxs(-1, xname, xtype)
        if (sectype == "rx"):
            _sxs.create_reactions_mode(xname, xtype, real, dev)
            _sxs.add_from_composition(self._index)
        else:
            _sxs.create_cross_section_mode(xname, xtype, real, dev)
            _sxs.add_from_composition(self._index)
        self._Sxs.append(_sxs)

    def updateSxs(self, xname, xtype, real, dev, sectype="rx"):
        for _sxs in self._Sxs:
            if (_sxs.name == xname) and (_sxs.xtype== xtype):
                if (sectype == "rx"):
                    _sxs.create_reactions_mode(xname, xtype, real, dev)
                    _sxs.update_from_composition(self._index)
                else:
                    _sxs.create_cross_section_mode(xname, xtype, real, dev)
                    _sxs.update_from_composition(self._index)

    @property
    def spectrum(self):
        return [(self._spectrumr[n], self._spectrumd[n])
                for n in range(self._lenspectrum.value)]
    @property
    def name(self):
        return self._name
    @property
    def energies(self):
        return self._energies

    @property
    def Sxs(self):
        return self._Sxs

    @property
    def numNuclide(self):
        return self._numNuclid
    @property
    def numEnergy(self):
        return self._numEnergy

class OpenBPSMaterial:

    def __init__(self, index=-1,
                  name=None, mass=None,
                  volume=None, power=None):
        self._initialize()
        if (index > -1):
            self._index = index
            self._read_material()
        else:
            self._index = index
            self._name = name
            self._mass = mass
            self._volume = volume
            self._power = power

        self._nuclides = []
    def _initialize(self):
        self._index = -1
        self._name = None
        self._mass = None
        self._volume = None
        self._power = None
        self._lennuclind = c_int(0)
        self._lennuclname = c_int(0)
        self._lennuclconc = c_int(0)
        self._nuclindexes = POINTER(c_int)()
        self._nuclnames = POINTER(c_char_p)()
        self._nuclconcr = POINTER(c_double)()
        self._nuclconcd = POINTER(c_double)()

    def _read_material(self):
        name = c_char_p()
        mass = c_double()
        volume = c_double()
        power = c_double()
        LIB.openbps_material_get_params_by_idx(self._index, name,
                                               mass, volume, power)
        self._name = name.value.decode()
        self._mass = mass.value
        self._power = power.value
        self._volume = volume.value
        return self

    def write_material(self):
        name_ptr = c_char_p(self._name.encode())
        LIB.openbps_material_set_params_by_idx(self._index, name_ptr,
                                               self._mass, self._volume,
                                               self._power)
    def write_nuclideConcentration(self, name, realconc, devconc):
        name_ptr = c_char_p(name.encode())
        LIB.openbps_material_set_nuclides_conc(self._index, name_ptr,
                                                   realconc, devconc)
    def add_material(self, name, mass, volume, power):
        name_ptr = c_char_p(name.encode())
        self._index=LIB.openbps_material_add(name_ptr,
                                               volume, power,
                                               mass)
    @staticmethod
    def get_material_number():
        n = c_int()
        LIB.openbps_materials_get_size(n)
        return n.value
    @property
    def nuclidIndexes(self):
        return [self._nuclindexes for n in range(self._lennuclind.value)]
    @property
    def nuclidConcentration(self):
        return [(self._nuclconcr[n], self._nuclconcd[n])
                for n in range(self._lennuclconc.value)]
    @property
    def nuclides(self):
        return [self._nuclnames[n] for n in range(self._lennuclname.value)]

    def get_nuclides(self):
        # Allocate memory for arguments that are written to
        LIB.openbps_material_get_idx_nuclides_by_idx(self._index,
                                                     self._nuclindexes,
                                                     self._lennuclind)
        print('index')
        LIB.openbps_material_get_conc_by_idx(self._index, self._nuclconcr,
                                             self._nuclconcd,
                                             self._lennuclconc)
        print('conc')
        LIB.openbps_material_get_nuclides_by_idx(self._index, self._nuclnames,
                                                 self._lennuclname)
        print('name')

    def add_nuclide(self, name, realconc, devconc):
        # Allocate memory for arguments that are written to
        name_ptr = c_char_p(name.encode())
        LIB.openbps_material_add_nuclide(self._index, name_ptr,
                                         realconc, devconc)
    def del_nuclide(self, name):
        name_ptr = c_char_p(name.encode())
        LIB.openbps_material_delete_nuclide(self._index, name_ptr)


class Material(_FortranObjectWithID):
    """Material stored internally.
    This class exposes a material that is stored internally in the OpenMC
    library. To obtain a view of a material with a given ID, use the
    :data:`openmc.lib.materials` mapping.
    Parameters
    ----------
    uid : int or None
        Unique ID of the material
    new : bool
        When `index` is None, this argument controls whether a new object is
        created or a view to an existing object is returned.
    index : int or None
         Index in the `materials` array.
    Attributes
    ----------
    id : int
        ID of the material
    nuclides : list of str
        List of nuclides in the material
    densities : numpy.ndarray
        Array of densities in atom/b-cm
    name : str
        Name of the material
    temperature : float
        Temperature of the material in [K]
    volume : float
        Volume of the material in [cm^3]
    """

    def __new__(cls, uid=None, new=True, index=None):
        #mapping = materials
##        self._name = ""
##        self._mass = -1
##        self._power = -1
##        self._volume = -1
        if index is not None:
            if new:
##                # Determine ID to assign
##                if uid is None:
##                    uid = max(mapping, default=0) + 1
##                else:
##                    if uid in mapping:
##                        raise AllocationError('A material with ID={} has already '
##                                              'been allocated.'.format(uid))

                _index = c_int32(index)
##                _dll.openmc_extend_materials(1, index, None)
##                index = index.value
                cls._index = _index.value
            else:
                _index = c_int32(index)
                cls._index = _index.value
##                index = mapping[uid]._index
        elif index == -1:
            # Special value indicates void material
            return None

##        if index not in cls.__instances:
##            instance = super(Material, cls).__new__(cls)
##            instance._index = index
##            if uid is not None:
##                instance.id = uid
##            cls.__instances[index] = instance

        return cls

    def read_material(self):
        name = c_char_p()
        mass = c_double()
        volume = c_double()
        power = c_double()
        LIB.openbps_material_get_params_by_idx(self._index, name,
                                               mass, volume, power)
        self._name = name.value.decode()
        self._mass = mass.value
        self._power = power.value
        self._volume = volume.value
        return self

    def write_material(self):
        name_ptr = c_char_p(self._name.encode())
        LIB.openbps_material_set_params_by_idx(self._index, name_ptr,
                                               self._mass, self._volume,
                                               self._power)
    def add_material(self, name, mass, volume, power):
        name_ptr = c_char_p(name.encode())
        self._index=LIB.openbps_material_add(name_ptr,
                                               volume, power,
                                               mass)
    @staticmethod
    def get_material_number():
        n = c_int()
        LIB.openbps_materials_get_size(n)
        return n.value
    @property
    def id(self):
        mat_id = c_int32()
        #_dll.openmc_material_get_id(self._index, mat_id)
        return mat_id.value

    @id.setter
    def id(self, mat_id):
        _dll.openmc_material_set_id(self._index, mat_id)

    @property
    def name(self):
        name = c_char_p()
        _dll.openmc_material_get_name(self._index, name)
        return name.value.decode()

    @name.setter
    def name(self, name):
        name_ptr = c_char_p(name.encode())
        _dll.openmc_material_set_name(self._index, name_ptr)

    @property
    def temperature(self):
        temperature = c_double()
        _dll.openmc_material_get_temperature(self._index, temperature)
        return temperature.value

    @property
    def volume(self):
        volume = c_double()
        try:
            _dll.openmc_material_get_volume(self._index, volume)
        except OpenMCError:
            return None
        return volume.value

    @volume.setter
    def volume(self, volume):
        _dll.openmc_material_set_volume(self._index, volume)

    @property
    def nuclides(self):
        return self._get_densities()[0]
        return nuclides

    @property
    def density(self):
      density = c_double()
      try:
          _dll.openmc_material_get_density(self._index, density)
      except OpenMCError:
          return None
      return density.value

    @property
    def densities(self):
        return self._get_densities()[1]
    def get_nuclides(self):
        # Allocate memory for arguments that are written to
        nuclides = POINTER(c_char_p)()
        indexes = POINTER(c_int)()
        realconcentations = POINTER(c_double)()
        devconcentations = POINTER(c_double)()
        LIB.openbps_material_get_nuclides_by_idx(self._index, nuclides)
        LIB.openbps_material_get_idx_nuclides_by_idx(self._index, indexes)
        LIB.openbps_material_get_conc_by_idx(self._index, realconcentations,
                                             devconcentations)
        return nuclides, indexes, realconcentations, devconcentations
    def add_nuclide(self, name, realconc, devconc):
        # Allocate memory for arguments that are written to
        name_ptr = c_char_p(name.encode())
        LIB.openbps_material_add_nuclide(self._index, name_ptr,
                                         realconc, devconc)
    def del_nuclide(self):
        name_ptr = c_char_p(name.encode())
        LIB.openbps_material_delete_nuclide(self._index, name_ptr)

    def _get_densities(self):
        """Get atom densities in a material.
        Returns
        -------
        list of string
            List of nuclide names
        numpy.ndarray
            Array of densities in atom/b-cm
        """
        # Allocate memory for arguments that are written to
        nuclides = POINTER(c_int)()
        densities = POINTER(c_double)()
        n = c_int()

        # Get nuclide names and densities
        _dll.openmc_material_get_densities(self._index, nuclides, densities, n)

        # Convert to appropriate types and return
        nuclide_list = [Nuclide(nuclides[i]).name for i in range(n.value)]
        density_array = as_array(densities, (n.value,))
        return nuclide_list, density_array

    def add_nuclide(self, name, density):
        """Add a nuclide to a material.
        Parameters
        ----------
        name : str
            Name of nuclide, e.g. 'U235'
        density : float
            Density in atom/b-cm
        """
        _dll.openmc_material_add_nuclide(self._index, name.encode(), density)

    def set_density(self, density, units='atom/b-cm'):
        """Set density of a material.
        Parameters
        ----------
        density : float
            Density
        units : {'atom/b-cm', 'g/cm3'}
            Units for density
        """
        _dll.openmc_material_set_density(self._index, density, units.encode())

    def set_densities(self, nuclides, densities):
        """Set the densities of a list of nuclides in a material
        Parameters
        ----------
        nuclides : iterable of str
            Nuclide names
        densities : iterable of float
            Corresponding densities in atom/b-cm
        """
        # Convert strings to an array of char*
        nucs = (c_char_p * len(nuclides))()
        nucs[:] = [x.encode() for x in nuclides]

        # Get numpy array as a double*
        d = np.asarray(densities)
        dp = d.ctypes.data_as(POINTER(c_double))

        _dll.openmc_material_set_densities(self._index, len(nuclides), nucs, dp)