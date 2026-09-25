# Unitree Go2 MuJoCo model

This directory contains the Go2 MJCF model and OBJ meshes from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2).
The files are included so the Phoenix sim2sim gate can load the same model from
a clean clone. See [PROVENANCE.md](PROVENANCE.md) for the upstream revision and
file hashes. The model's BSD 3-Clause license is in [LICENSE](LICENSE).

Load `scene.xml` for the gate. It includes `go2.xml` and the floor geometry.
