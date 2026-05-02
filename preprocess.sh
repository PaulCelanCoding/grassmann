cd data/n3dv/flame_steak
for i in $(seq -w 0 20); do
    mkdir -p cam${i}/images
    ffmpeg -i cam${i}.mp4 -vf "fps=30" cam${i}/images/%04d.png
    # rename to 0-indexed
    cd cam${i}/images && ls *.png | awk 'BEGIN{i=0}{printf "mv %s %04d.png\n", $0, i++}' | sh
    cd ../..
done
cd ../..