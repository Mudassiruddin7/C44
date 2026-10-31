"""C44 / W2V: videos rendered from AI-generated 3D worlds vs. video forensics.

Cells of the factorial design (all clips share one spec, see config.SPEC):
  RC  real scene, camera capture        (DL3DV raw video)
  RR  real scene, 3DGS render           (DL3DV-GS-960P, rendered along a motion-matched path)
  GR  generated world, 3DGS render      (FlashWorld, conditioned on the RC first frame + RC camera path)
  GV  generated video, video diffusion  (reference cell)
"""

__version__ = "0.1.0"
