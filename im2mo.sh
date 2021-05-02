set filename $argv[1]
set fps $argv[2]
ffmpeg -i output/images/$filename/image-%05d.png -vcodec libx264 -r $fps output/movies/$filename.mp4
