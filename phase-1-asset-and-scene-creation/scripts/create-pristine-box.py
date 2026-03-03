import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg

# Configuration for a deformable cube that resembles a cardboard box
cardboard_box_cfg = DeformableObjectCfg(
    prim_path="/World/PristineBox",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.3, 0.4, 0.3), # Dimensions of a small box in meters
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            rest_offset=0.0,
            contact_offset=0.005
        ),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            youngs_modulus=5e5,     # A moderately stiff material
            poissons_ratio=0.2,     # Lower ratio, less lateral deformation
            damping_scale=1.0,
            elasticity_damping=0.005
        )
    )
)

# Create the deformable object in the scene
pristine_box = DeformableObject(cfg=cardboard_box_cfg)