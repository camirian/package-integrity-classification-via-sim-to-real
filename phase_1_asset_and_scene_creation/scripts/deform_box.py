import omni.timeline
from pxr import Usd, UsdGeom

# --- Assume the pristine_box from Step 1 exists ---

# Create a heavy rigid sphere to act as the damaging object
rigid_sphere_cfg = sim_utils.SphereCfg(
    prim_path="/World/DamagingSphere",
    radius=0.1,
    mass=50.0 # Heavy mass to cause deformation
)
rigid_sphere_cfg.spawn(translation=(0.0, 0.0, 0.5))

# Let the simulation run for a short duration
timeline = omni.timeline.get_timeline_interface()
timeline.play()
# This part requires manual intervention or a more complex script with callbacks.
# For this guide, assume you manually step the simulation forward until the
# sphere hits the box and deforms it, then you pause it.
# await omni.kit.app.get_app().next_update_async() # In a real script

# --- Manually pause the simulation in the UI at the desired deformation ---

# Save the deformed mesh as a new asset
stage = omni.usd.get_context().get_stage()
deformed_prim = stage.GetPrimAtPath("/World/PristineBox")
# Create a new USD file to save the deformed geometry
omni.kit.commands.execute('CreateUsd',
    dest_path='\\wsl$\Ubuntu\home\caaren\package_integrity_classification_via_sim-to-real\phase_1_asset_and_scene_creation\assets\box_dented.usd',
    stage_identifier=deformed_prim.GetPath().pathString)

# Repeat with more mass/higher drop for a 'box_crushed.usd'