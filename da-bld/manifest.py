from collections import namedtuple

Item = namedtuple('Item', ['package_name', 'operation'])

__all__ = (
    'MANIFEST'
)

MANIFEST = [
    Item('bld_lattice_library', lambda layout: layout.operator(
        'designauto.lattice_library', text="晶格库")),
    Item('bld_lattice_design', lambda layout: layout.operator(
        'designauto.lattice_design_voxel_structure', text="体素晶格设计")),
    Item('bld_lattice_design', lambda layout: layout.operator(
        'designauto.lattice_design_field_driven_voxel', text="场引导体素晶格设计")),
    Item('bld_lattice_design', lambda layout: layout.operator(
        'designauto.lattice_design_voronoi_structure', text="三维 Voronoi 晶格设计")),
    Item('bld_lattice_design', lambda layout: layout.operator(
        'designauto.lattice_design_hexahedron_structure', text="六面体晶格设计")),
    Item('bld_lattice_design', lambda layout: layout.operator(
        'designauto.lattice_design_tpms_structure', text="TPMS晶格设计")),
    Item('bld_lattice_design', lambda layout: layout.operator(
        'designauto.lattice_design_field_driven_tpms', text="场引导TPMS晶格设计")),
    
    Item('---', lambda layout: layout.separator()),
    
    Item('bld_quasistatic_simulation', lambda layout: layout.operator(
        'designauto.quasistatic_simulation_fem_tetrahedral', text="FEM仿真")),

    Item('bld_quasistatic_simulation', lambda layout: layout.operator(
        'designauto.quasistatic_simulation_cbn_polyhedron', text="CBN仿真")),

    Item('---', lambda layout: layout.separator()),
    
    Item('bld_quasistatic_simulation', lambda layout: layout.operator(
        'designauto.texture_3d_surface_hole', text="圆孔纹理")),

    Item('---', lambda layout: layout.separator()),

    Item('bld_top_density_field_display', lambda layout: layout.operator(
        'designauto.top_density_field_display', text="密度场展示")),
]
