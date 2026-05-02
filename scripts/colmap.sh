mkdir colmap_input
for d in data/n3dv/flame_steak/cam*; do
    cp "$d/images/0000.png" "colmap_input/$(basename $d).png"
done
colmap automatic_reconstructor --workspace_path colmap_workspace \
                                --image_path colmap_input
cp colmap_workspace/sparse/0/points3D.txt data/n3dv/flame_steak/points3D.txt