import gmsh
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import numpy as np



def gmsh_sphere(model: gmsh.model, name: str) -> gmsh.model:
    """Create a Gmsh model of a sphere and tag sub entitites
    from all co-dimensions (peaks, ridges, facets and cells).

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a sphere mesh added.

    """
    model.add(name)
    model.setCurrent(name)
    sphere = model.occ.addSphere(0, 0, 0, 1, tag=1)


    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical tag for sphere
    surfaces = model.getBoundary([(3, sphere)], oriented=False, recursive=False)
    surface_tags = [s[1] for s in surfaces if s[0] == 2]
    model.add_physical_group(2, surface_tags, tag=1)
    

    # Generate the mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
    model.mesh.generate(dim=2)
    return model

def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """Create a DOLFINx from a Gmsh model and output to file.

    Args:
        comm: MPI communicator top create the mesh on.
        model: Gmsh model.
        name: Name (identifier) of the mesh to add.
        filename: XDMF filename.
        mode: XDMF file mode. "w" (write) or "a" (append).
    """
    mesh_data = gmshio.model_to_mesh(model, comm, rank=0)
    mesh_data[0].name = name

    with XDMFFile(mesh_data[0].comm, filename, mode) as file:
        mesh_data[0].topology.create_connectivity(2, 2)
        mesh_data[0].topology.create_connectivity(1, 2)
        mesh_data[0].topology.create_connectivity(0, 2)
        file.write_mesh(mesh_data[0])


def write_sphere():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    model = gmsh_sphere(model, "sphere")
    model.setCurrent("sphere")
    create_mesh(MPI.COMM_SELF, model, "sphere", f"out_gmsh/sphere.xdmf", "w")
    
def gmsh_hemisphere(model: gmsh.model, name: str) -> gmsh.model:
    model.add(name)
    model.setCurrent(name)
    sphere = model.occ.addSphere(0, 0, 0, 1, tag=1)
    cut_box = gmsh.model.occ.addBox(-2, -2, -2, 4, 4, 2)
    hemisphere_tags, hemisphere_map = gmsh.model.occ.cut([(2, s) for s in [sphere]], [(3, cut_box)], removeObject=True)



    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical tag for sphere
    surface_tags = [s[1] for s in hemisphere_tags if s[0] == 2]
    model.add_physical_group(2, surface_tags, tag=1)
    

    # Generate the mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
    model.mesh.generate(dim=2)
    return model

def write_hemisphere():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    model = gmsh_hemisphere(model, "hemisphere")
    model.set_current("hemisphere")
    create_mesh(MPI.COMM_SELF, model, "hemisphere", f"out_gmsh/hemisphere.xdmf", "w")

write_sphere()
