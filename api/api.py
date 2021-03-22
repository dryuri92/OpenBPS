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
import os
import sys

if sys.platform == 'darwin':
    _suffix = 'dylib'
else:
    _suffix = 'so'

nameoflib = "/mnt/c/virtual/env/OpenBPS/build/lib/liblibopenbps.{}"



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

import sys
sys.path.append("./api")
from api import *
#init()
#material = OpenBPSMaterial(index=0)
#material.get_nuclides()